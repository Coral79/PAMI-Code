"""Vendored from MotionStreamer models/tae_part.py - ONLY get_part_indices_272_smplh
(pure index bookkeeping). The Causal_TAE_Part class itself is NOT vendored - pami has its own
network definitions."""


def get_part_indices_272_smplh():
    """
    基于SMPLH的22关节，返回6个身体部位的精确索引

    272维结构:
    [0:2]     - Root XZ速度
    [2:8]     - Heading 6D旋转
    [8:74]    - 22关节位置 (22×3)
    [74:140]  - 22关节速度 (22×3)
    [140:272] - 22关节旋转 (22×6)
    """

    # SMPLH关节分组
    torso_joints = [0, 3, 6, 9, 12, 15]      # 6个关节
    left_arm_joints = [13, 16, 18, 20]        # 4个关节
    right_arm_joints = [14, 17, 19, 21]       # 4个关节
    left_leg_joints = [1, 4, 7, 10]           # 4个关节
    right_leg_joints = [2, 5, 8, 11]          # 4个关节

    def get_joint_features(joint_ids):
        """获取指定关节的 位置+速度+旋转 的所有索引"""
        indices = []

        # 位置特征 [8:74]
        for jid in joint_ids:
            start = 8 + jid * 3
            indices.extend([start, start+1, start+2])

        # 速度特征 [74:140]
        for jid in joint_ids:
            start = 74 + jid * 3
            indices.extend([start, start+1, start+2])

        # 旋转特征 [140:272]
        for jid in joint_ids:
            start = 140 + jid * 6
            indices.extend(range(start, start+6))

        return indices

    # 构建6个部位的索引
    part_indices = [
        list(range(0, 8)),                     # Part 0: Root (8维)
        get_joint_features(torso_joints),      # Part 1: Torso (72维)
        get_joint_features(left_arm_joints),   # Part 2: Left Arm (48维)
        get_joint_features(right_arm_joints),  # Part 3: Right Arm (48维)
        get_joint_features(left_leg_joints),   # Part 4: Left Leg (48维)
        get_joint_features(right_leg_joints),  # Part 5: Right Leg (48维)
    ]

    return part_indices


