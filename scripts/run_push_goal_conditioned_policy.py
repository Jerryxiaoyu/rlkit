import argparse
import torch

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout,multitask_dict_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
import gym
import multiworld


def get_env(render=False):
    from multiworld.core.image_raw_env import ImageRawEnv
    from multiworld.envs.pybullet.cameras import jaco2_push_top_view_camera
    wrapped_env = gym.make("Jaco2PushPrimitiveDiscOneXYEnv-v0" ,isRender = render,
    isImageObservation= False,

   maxActionSteps=10,

   isRandomGoals=False,
   isIgnoreGoalCollision=False,
   fixed_objects_goals=(-0, -0.3,
                         ),  # shape (3*n,)

   vis_debug=True)

    env = ImageRawEnv(
            wrapped_env,
            init_camera=jaco2_push_top_view_camera,
            heatmap=True,
            normalize=False,
            reward_type='wrapped_env',
            goal_in_image_dict_key = 'valid_depth_heightmap',
            image_achieved_key='valid_depth_heightmap',
            recompute_reward = False,
        )
    return env

def simulate_policy(args):


    data = torch.load(args.file)
    policy = data['evaluation/policy']
    env = get_env(True)

#    env = gym.make(env.spec.id, isRender=True, isRenderGoal= True,)

    print("Policy and environment loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    paths = []
    while True:
        paths.append(multitask_dict_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key='heatmap',
            desired_goal_key='heatmap',
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        help='path to the snapshot file',
                        default='/home/drl/PycharmProjects/JerryRepos/rlkit/data/06-09-jaco-push-dqn/06-09-jaco_push_dqn_2020_06_09_14_22_36_0000--s-46168/params.pkl')
    parser.add_argument('--H', type=int, default=30,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    multiworld.register_pybullet_envs()
    simulate_policy(args)
