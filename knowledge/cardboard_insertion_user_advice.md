# Cardboard insertion user advice

This file records user observations that should guide future cardboard
insertion experiments.

## Current target

- The next performance target is not just 5 mm insertion distance.
- Aim for stable 3 mm-level distance between the purple finger marker and the
  lime cabinet target.
- Good results must still be fixed-action-bias free and should be shown with
  evaluation video.

## EEF orientation hint

The user observed from Spacemouse trials that a successful insertion posture
likely needs:

- EEF local x axis aligned with world +x as a strong constraint.
- In addition, the useful insertion posture may have the EEF frame rotated
  around its own x axis by roughly 45 to 65 degrees.

Do not treat this 45 to 65 degree range as proven yet. Treat it as a high-value
hypothesis to visualize, measure on successful rollouts, and test with reward
or curriculum variants.

## Gripper opening hint

The gripper opening seen in the user's Spacemouse success example is important.
Rewarding the gripper qpos to stay near that open configuration should apply
throughout the episode, including before and while reaching the first waypoint.

## Video filenames

When creating evaluation videos intended to be shown to the user, or when the
user asks to see a video, use Japanese filenames by default. This makes the
purpose of the artifact easier to recognize in chat and file browsers. For
example:

- `こちらをご参照ください.mp4`
- `現在のベスト結果_近接視点.mp4`
- `現在のベスト結果_俯瞰視点.mp4`

Internal throwaway logs and batch experiment directories can still use ASCII
names when that is more convenient for scripts.
