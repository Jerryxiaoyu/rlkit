"""
Run DQN on grid world.
"""
import os, inspect
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
print(project_dir)
os.sys.path.insert(0, project_dir)

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy, PushPrimitiveWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, PushNetEpsilonGreedy, PushNetPrimitiveEpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.dqn.push_dqn import PushDQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector,GoalConditionedDictPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
import multiworld
from rlkit.torch.push_net.pushnet import PushNet
from rlkit.policies.pushnet_policy import PushNetArgmaxDiscretePolicy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.policies.push_random_policy import HeuristicPushPolicy
from multiworld.envs.pybullet.cameras import jaco2_push_top_view_camera



# def get_env(render=False):
#     from multiworld.core.image_raw_env import ImageRawEnv
#
#     wrapped_env = gym.make("Jaco2PushPrimitiveDiscOneXYEnv-v0" ,isRender = render,
#     isImageObservation= False,
#
#     reward_mode = 0, ## 0 for dense, 1 for sparse
#     maxActionSteps=10,
#
#    isRandomGoals=False,
#    isIgnoreGoalCollision=False,
#    fixed_objects_goals=(-0, -0.3,
#                          ),  # shape (3*n,)
#
#    vis_debug=False)
#
#     env = ImageRawEnv(
#             wrapped_env,
#             init_camera=jaco2_push_top_view_camera,
#             heatmap=True,
#             normalize=False,
#             reward_type='wrapped_env',
#             goal_in_image_dict_key = 'valid_depth_heightmap',
#             image_achieved_key='valid_depth_heightmap',
#             recompute_reward = False,
#         )
#     return env


def experiment(variant):
    import gym
    from torch import nn as nn
    from rlkit.exploration_strategies.base import \
        PolicyWrappedWithExplorationStrategy, PushPrimitiveWithExplorationStrategy
    from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, PushNetEpsilonGreedy, \
        PushNetPrimitiveEpsilonGreedy
    from rlkit.policies.argmax import ArgmaxDiscretePolicy
    from rlkit.torch.dqn.dqn import DQNTrainer
    from rlkit.torch.dqn.push_dqn import PushDQNTrainer
    from rlkit.torch.networks import Mlp
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
    from rlkit.launchers.launcher_util import setup_logger
    from rlkit.samplers.data_collector import MdpPathCollector, GoalConditionedDictPathCollector
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
    import multiworld
    from rlkit.torch.push_net.pushnet import PushNet
    from rlkit.policies.pushnet_policy import PushNetArgmaxDiscretePolicy
    from rlkit.launchers.launcher_util import run_experiment
    from rlkit.policies.push_random_policy import HeuristicPushPolicy
    from multiworld.envs.pybullet.cameras import jaco2_push_top_view_camera
    from rlkit.core import logger

    logger.set_aws_mode(variant['bucket_path'])
    def get_env(render=False, **env_kwargs):
        from multiworld.core.image_raw_env import ImageRawEnv

        wrapped_env = gym.make("Jaco2PushPrimitiveDiscOneXYEnv-v0", isRender=render,
                               isImageObservation=False,

                               **env_kwargs,
                               )

        env = ImageRawEnv(
            wrapped_env,
            init_camera=jaco2_push_top_view_camera,
            heatmap=True,
            normalize=False,
            reward_type='wrapped_env',
            goal_in_image_dict_key='valid_depth_heightmap',
            image_achieved_key='valid_depth_heightmap',
            recompute_reward=False,
        )
        return env

    expl_env = get_env(render=False, **variant['env_kwargs'])
    eval_env = get_env(render=False, **variant['env_kwargs'])
    #obs_dim = expl_env.observation_space.low.size

    action_space = (16,50,50)


    observation_key = 'heatmap' #variant['observation_key']
    desired_goal_key = 'heatmap' #variant['desired_goal_key']

    qf = PushNet(use_cuda=True, n_batch = variant['algorithm_kwargs']['batch_size'])
    target_qf = PushNet(use_cuda=True, n_batch = variant['algorithm_kwargs']['batch_size'])


    qf_criterion = nn.SmoothL1Loss(reduce=False)
    eval_policy = PushNetArgmaxDiscretePolicy(qf)

    #PushNetArgmaxDiscretePolicy(qf)
    # expl_policy = PolicyWrappedWithExplorationStrategy(
    #     PushNetEpsilonGreedy(action_space, prob_random_action=0.3),
    #     eval_policy,
    # )
    primitive_policy = HeuristicPushPolicy(
        dict(
            camera_params=jaco2_push_top_view_camera,
            num_objects=1,
            cspace_high=(0.2, -0.2, 0.154),
            cspace_low=(-0.2, -0.6, 0.154),
            mask_bias=3,
            heightmap_resolution=0.008
        )
    )
    expl_policy = PushPrimitiveWithExplorationStrategy(
        eval_policy,
        primitive_policy,
         **variant['primitive_kwargs'],


    )
    eval_path_collector = GoalConditionedDictPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedDictPathCollector(
        expl_env,
        expl_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    trainer = PushDQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        ignore_keys =['observations','next_observations'],
        **variant['trainer_kwargs']
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        **variant['replay_buffer_kwargs']
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    observation_key = 'heatmap'
    desired_goal_key = 'heatmap'

    variant = dict(
        env_kwargs=dict(
            reward_mode=1,  ## 0 for dense, 1 for sparse
            maxActionSteps=10,
            obj_name_list=['b_L1'],
            isRandomGoals=False,
            isIgnoreGoalCollision=False,
            fixed_objects_goals=(-0, -0.3,  ),  # shape (3*n,)

            vis_debug=False
        ),
        primitive_kwargs = dict(
            init_prob=0.5,
            decay_rate=0.97,
        ),

        observation_key = observation_key,
        desired_goal_key = desired_goal_key,
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_kwargs=dict(
            max_size=100000 ,
            fraction_goals_rollout_goals=1,  # 0.2 equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
            observation_key='heatmap',
            desired_goal_key='state_desired_goal',
            achieved_goal_key='state_observation',
            internal_keys=['changed_object'],
            goal_keys=None,
        ),
        algorithm_kwargs=dict(
            num_epochs= 100 ,
            num_eval_steps_per_epoch= 100  , # 100
            num_train_loops_per_epoch = 1, #1
            num_trains_per_train_loop= 200 ,#1000  200
            num_expl_steps_per_train_loop= 500,#500
            min_num_steps_before_training= 0,
            max_path_length= 10,
            batch_size= 6,#2
        ),
        trainer_kwargs=dict(
            discount=0.5,
            learning_rate=3E-4,
            target_update_period=200,
            soft_target_tau=1,

            reward_mode=1,  # for sparse
        ),
        bucket_path = 'jerry-castle/castle_q_learning/results'  #None
    )

    #setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    #experiment(variant)


    mode = 'here_no_doodad'#'here_no_doodad'
    exp_prefix = 'jaco_push_dqn'

    ptu.set_gpu_mode(True)


    run_experiment(
        experiment,
        exp_prefix=exp_prefix,
        mode=mode,
        variant=variant,
        region='us-east-2',
        use_gpu=True,
        verbose=True,
  )