from .DataManager import DataManager


class DataManagerVec(DataManager):
    """Data manager with vectorization."""

    def reset(self):
        """Reset."""
        self.all_data_seq_list = [{} for env_idx in range(self.env.unwrapped.num_envs)]

    def append_single_data(self, key, data_list):
        """Append a single data to the data sequence."""
        for all_data_seq, data in zip(self.all_data_seq_list, data_list):
            if key not in all_data_seq:
                all_data_seq[key] = []
            all_data_seq[key].append(data)

    def get_single_data(self, key, time_idx):
        """Get a single data from the data sequence."""
        data_list = []
        for all_data_seq in self.all_data_seq_list:
            data = all_data_seq[key][time_idx]
            data_list.append(data)
        return data_list

    def get_data_seq(self, key):
        """Get a data sequence."""
        data_seq_list = []
        for all_data_seq in self.all_data_seq_list:
            data_seq = all_data_seq[key]
            data_seq_list.append(data_seq)
        return data_seq_list

    def calc_rel_data(self, key):
        """Calculate relative data."""
        return [
            super(DataManagerVec, self).calc_rel_data(key, all_data_seq)
            for all_data_seq in self.all_data_seq_list
        ]

    def save_data(self, filename_list):
        """Save data."""
        for all_data_seq, filename in zip(self.all_data_seq_list, filename_list):
            if filename is None:
                continue

            super(DataManagerVec, self).save_data(
                filename, all_data_seq, self.meta_data, increment_episode_idx=False
            )

        self.episode_idx += 1

    def load_data(self, filename_list):
        raise NotImplementedError(
            f"[{self.__class__.__name__}] This method is not supported."
        )
