from .DataKey import DataKey


class EnvDataMixin:
    @property
    def command_keys_for_step(self):
        return [DataKey.COMMAND_JOINT_POS]

    @property
    def measured_keys_to_save(self):
        return [
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.MEASURED_EEF_POSE,
            DataKey.MEASURED_EEF_WRENCH,
        ]

    @property
    def command_keys_to_save(self):
        return [
            DataKey.COMMAND_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
            DataKey.COMMAND_EEF_POSE,
        ]
