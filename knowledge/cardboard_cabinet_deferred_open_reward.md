# Deferred cardboard cabinet open reward

## Context

The branch temporarily tested a new reward/success definition for `MyDualCardboardCabinet-v1` that measures how far the inner cardboard box opens. The idea worked as an experiment target, but it is deferred for now so the current branch can preserve the known-good behavior and evaluation video.

## Deferred idea

Add an inner-box opening metric:

- Compute the cabinet-local open direction in world coordinates.
- Measure displacement of `cardboard_inner_box` from its initial world position.
- Project that displacement onto the open direction.
- Clamp negative displacement to zero.
- Normalize by an open goal distance, tested as `0.12`.

Useful constants from the deferred experiment:

```python
OPEN_STARTED_DISTANCE = 0.005
INNER_BOX_OPEN_GOAL_DISTANCE = 0.12
INNER_BOX_STATIC_LINEAR_VEL = 0.05
INNER_BOX_STATIC_ANGULAR_VEL = 1.0
```

The deferred `evaluate()` fields were:

```python
inner_box_open_amount
inner_box_open_fraction
open_enough
inner_box_static
success = open_enough & inner_box_static
```

The deferred dense reward was:

```python
reaching_reward = tcp_to_target_reward
open_reward = 2.0 * inner_box_open_fraction

if inner_box_open_amount >= OPEN_STARTED_DISTANCE:
    reaching_reward = 2.0

if open_enough:
    open_reward = 3.0

reward = reaching_reward + open_reward
reward[success] = 5.0
normalized_reward = reward / 5.0
```

## Why deferred

This changes the task objective from only reaching/gripper-shaping into an explicit drawer-opening objective. Keep it out of the current training branch until the existing valuable policy/video is preserved.

The currently valuable video to preserve is:

```text
tmp_note/eval_videos/cardboard_latest/PPO_cardboard_latest_slot_open_ckpt201.mp4
```
