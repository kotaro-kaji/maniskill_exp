# Cardboard arm qpos return handoff

## Current direction

For cardboard drawer return experiments, keep the reward direction centered on
arm-only qpos return unless the user explicitly changes the research direction.

The TCP return metric may remain useful as a diagnostic signal, but it should
not replace the training reward by default. Prior TCP-return trials showed that
drawer opening can stay strong while TCP return remains insufficient, so reward
changes should be judged against both final drawer opening and final arm return
quality.

## Evaluation preference

Use final-state metrics when reporting return behavior:

- `final_open_success` or equivalent final drawer-open condition.
- final arm qpos return score/error.
- outer-box shift or stability metric.
- TCP return error only as a diagnostic metric.

Older reports may include intermediate or reached-at-any-time success counters.
Do not treat those as sufficient evidence for final return success.

## Related durable context

- `drawer_reward_lessons.md`: Drawer opening reward lessons and 12 cm success
  target context.
- `../../outputs_to_user/cardboard_drawer_key_commits.html`: historical commit
  summary for drawer return task setup.
- `../../outputs_to_user/cardboard_tcp_return_status.html`: prior TCP-return
  experiment status.
