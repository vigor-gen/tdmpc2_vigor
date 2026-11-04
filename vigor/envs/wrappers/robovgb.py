import copy
import os
from typing import Union, Optional

import cv2
import gymnasium as gym
import hydra
import numpy as np
import robosuite as suite
from robosuite.utils.mjmod import LightingModder, TextureModder

from vigor.envs.wrappers import preset_customization
from vigor.envs.wrappers.custom_xml import CustomMujocoXML, XMLTextureModder

ALL_ROBOTS = list(suite.ALL_ROBOTS)

class RoboVGBWrapper(gym.Wrapper):
    def __init__(self, env, task, seed, image_size, gen_type=None, scene_id=None):
        super().__init__(env)
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize(config_path="../cfgs", version_base="1.3"):
            self._robovgb_cfg = hydra.compose(config_name="robovgb")
        gen_type = gen_type or self._robovgb_cfg.gen_type
        assert gen_type in ["train", "eval-easy", "eval-hard", "eval-extreme"], f"{gen_type} is not supported."
        self._gen_type = gen_type
        if self._gen_type == "train":
            self._randomize_color = self._randomize_lighting = False
            self._moving_light = False
            self._except_robot = True
            self._video_background = False
        elif self._gen_type == "eval-easy":
            self._randomize_color = self._randomize_lighting = True
            self._moving_light = False
            self._except_robot = True
            self._video_background = False
        elif self._gen_type == "eval-hard":
            self._randomize_color = self._randomize_lighting = True
            self._moving_light = True
            self._except_robot = False
            self._video_background = False
        elif self._gen_type == "eval-extreme":
            self._randomize_color = self._randomize_lighting = True
            self._moving_light = True
            self._except_robot = True
            self._video_background = True

        scene_id = scene_id or 0
        assert scene_id in range(10), "scene_id range: 0 ~ 9"
        self._scene_id = scene_id
        np.random.seed(seed)
        self._random_state = np.random.RandomState(seed)
        self._image_size = image_size
        self._obs_modality = "rgb"
        self._use_depth = False
        self._video_buf = self._load_video(self._gen_type, self._image_size, self._image_size) if self._video_background else None
        self._current_video_frame = 0

        custom_reset_config = preset_customization.get_custom_reset_config(
            task=task, gen_type=self._gen_type, scene_id=self._scene_id
        )
        if custom_reset_config is not None:
            if self._video_background:
                custom_reset_config["custom_texture"]["tex_candidate"]["floor"] = custom_reset_config["custom_texture"]["tex_candidate"]["wall"] = "black"
            self._initialize_xml_modder(custom_reset_config.get("custom_texture", None), task)
            if custom_reset_config.get("custom_texture") is not None:
                self._reset_xml_next = True
        else:
            self._reset_xml_next = False
        self.is_first_episode = True

    def _initialize_modders(self):
        self._modders = []
        if self._randomize_color:
            self._robovgb_cfg.color.geom_names = tuple(
                geom for geom in filter(lambda x: x != None, self.env.sim.model.geom_names) if "robot" not in geom and "gripper" not in geom)
            self._tex_modder = TextureModder(
                sim=self.env.sim, random_state=self._random_state, **self._robovgb_cfg.color
            )
            self._modders.append(self._tex_modder)

        if self._randomize_lighting:
            self._light_modder = LightingModder(
                sim=self.env.sim,
                random_state=self._random_state,
                **self._robovgb_cfg.light,
            )
            self._modders.append(self._light_modder)

        if self._moving_light:
            self.env.sim.model.light_pos[:, 2] = 4
            self._origin_light_pos = copy.deepcopy(self.env.sim.model.light_pos)
            self.env.sim.model.light_castshadow = 1
            self.env.sim.model.light_directional = 0
            self._moving_light_step = 0.03
            self.light_pos_range = [self.env.sim.model.light_pos[:, 1] - 3, self.env.sim.model.light_pos[:, 1] + 3]

    def _initialize_xml_modder(self, custom_texture: Optional[Union[str, dict]], task: str):
        if custom_texture is not None:
            config = custom_texture.copy()
            if custom_texture.get("tex_to_change", None) is None:
                config["tex_to_change"] = preset_customization.DEFAULT_TASK_TEXTURE_LIST[task]
            self._secant_texture_modders = XMLTextureModder(**config)
        else:
            self._secant_texture_modders = None

    def _load_video(self, gen_type, image_height, image_width):
        current_dir = os.path.dirname(__file__)
        video_dir = os.path.join(current_dir, "../data", "robovgb", gen_type)
        video_path = os.path.join(video_dir, f"video{self._scene_id}.mp4")
        cap = cv2.VideoCapture(video_path)
        self._video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty((self._video_len, image_height, image_width, 3),
                       np.dtype("uint8"))
        i, ret = 0, True
        while (i < self._video_len and ret):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (image_height, image_width))
            buf[i] = frame
            i += 1
        cap.release()
        return np.moveaxis(buf, -1, 1)

    def _reformat_obs(self, obs_dict, disable_channel_first=False):
        rgb_obs = {}
        state_obs = []
        for name, observation in obs_dict.items():
            if name.endswith("_image") and "rgb" in self._obs_modality:
                view_name = name[:-6]
                if self._use_depth:
                    depth_obs = obs_dict[view_name + "_depth"]
                    observation = np.concatenate([observation, depth_obs[:, :, np.newaxis]], axis=2)
                observation = np.flipud(observation)
                if not disable_channel_first:
                    observation = np.transpose(observation, (2, 0, 1))
                rgb_obs[view_name] = observation
            elif name.endswith("state") and "state" in self._obs_modality:
                state_obs.append(observation)
        if "rgb" in self._obs_modality:
            if len(rgb_obs.keys()) == 1:
                rgb_obs = list(rgb_obs.values())[0]
            reformatted = rgb_obs
            if self._video_background:
                fixedRGB = np.array([0, 0, 0], dtype="uint8")
                reformatted = reformatted.transpose(1, 2, 0)
                idx = np.where((reformatted[:][:] == fixedRGB).all(axis=2), np.array(1, dtype=np.uint8),
                               np.array(0, dtype=np.uint8))
                idx = np.array([idx, idx, idx])
                reformatted = reformatted.transpose(2, 0, 1)
                reformatted = (1 - idx) * reformatted + idx * self._video_buf[self._current_video_frame % self._video_len, :,
                                              :, :]
        if "state" in self._obs_modality:
            reformatted = np.concatenate(state_obs)
        return reformatted

    def reset(self):
        self._current_video_frame = 0
        # reset from xml should only be called if a different texture/xml is requested
        if self._reset_xml_next:
            mujoco_xml = CustomMujocoXML.build_from_env(self)
            if self._secant_texture_modders is not None:
                self._secant_texture_modders.random_texture_change(mujoco_xml)
            self.env.reset_from_xml_string(mujoco_xml.to_string())
            self._reset_xml_next = False
        else:
            self.env.reset()

        # update sims
        if self.is_first_episode:
            self._initialize_modders()
            self.is_first_episode = False
        for modder in self._modders:
            modder.update_sim(self.env.sim)
            modder.randomize()
        if not self._except_robot:
            rgb = self._random_state.uniform(0, 1, size=3)
            for geom in filter(lambda x: x != None, self.env.sim.model.geom_names):
                if "robot" in geom or "gripper" in geom:
                    self._tex_modder.set_geom_rgb(geom, rgb)
        if self._moving_light:
            self.env.sim.model.light_pos = self._origin_light_pos
        self.env._reset_internal()
        self.env.sim.forward()
        return self._reformat_obs(self.env._get_observations(force_update=True))

    def step(self, action):
        self._current_video_frame += 1
        if self._moving_light:
            self.env.sim.model.light_pos[:, :] += self._moving_light_step
            y = self.env.sim.model.light_pos[:, 1]
            low, high = self.light_pos_range
            if np.any(y > high) or np.any(y < low):
                self._moving_light_step *= -1
        ob_dict, reward, done, info = self.env.step(action)
        info["success"] = self.env._check_success()
        return self._reformat_obs(ob_dict), reward, done, info

    def render(self, height=None, width=None, camera_id=0):
        image = cv2.flip(self.env.sim.render(height, width), 0)
        if self._video_background:
            fixedRGB = np.array([0, 0, 0], dtype="uint8")
            idx = np.where((image[:][:] == fixedRGB).all(axis=2), np.array(1, dtype=np.uint8),
                           np.array(0, dtype=np.uint8))
            idx = np.array([idx, idx, idx]).transpose(1, 2, 0)
            image = (1 - idx) * image + \
                idx * cv2.resize(self._video_buf[self._current_video_frame % self._video_len, :, :, :].transpose(1, 2, 0), (image.shape[:2]))
        return image

    def observation_spec(self):
        return self.env.observation_spec()

    @property
    def action_spec(self):
        return self.env.action_spec

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.env:
                    return self
                return result

            return hooked
        else:
            return orig_attr