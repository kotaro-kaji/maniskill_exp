# Cardboard insertion policy notes

## No fixed action offsets

Do not use evaluation-time fixed action offsets for cardboard insertion results.
In particular, do not add constants such as `action[1] += ...` or
`action[4] += ...` to a learned policy action at rollout/evaluation time.

Reason: fixed action offsets make the result ambiguous. They mix the learned
policy with a hand-tuned controller correction, so the reported behavior is no
longer the policy's actual performance.

Allowed actions:

- Change task geometry, robot/controller limits, reward terms, or training
  procedure explicitly in code.
- Train or fine-tune a policy under those explicit settings.
- Evaluate the policy action directly, with only normal environment action
  clipping.
- Use videos and scalar metrics from no-offset evaluation as the main evidence.

Disallowed actions:

- Reporting a checkpoint result that depends on a hand-tuned runtime action
  offset.
- Adding temporary offset arguments to training or evaluation scripts.
- Keeping old offset-search scripts or videos as current evidence.

## Current useful context

The current best no-offset cardboard rollout uses:

- `robot_init_noise_scale=0.0`
- `pd_joint_delta_pos`
- inner box wall thickness `2 mm`
- insert target local z raised by `1 mm`
- relaxed box shift success threshold `7 mm`

Current no-offset video paths are documented in
`tmp_note/cardboard_insertion_progress.md`.

## Next target

The next target is stable 3 mm-level insertion distance. Keep using no fixed
runtime action offsets. User-provided orientation and gripper-opening hints are
tracked in `insertion_user_advice.md`.
