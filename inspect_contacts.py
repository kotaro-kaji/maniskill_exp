import numpy as np
import gymnasium as gym

import mani_skill.envs
import task_pushcube_beatiful


def main():
    env = gym.make(
        "MyPushCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
        sim_backend="cpu",
    )
    obs, _ = env.reset(seed=0)

    # 少しだけ動かして接触を発生させる
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(120):
        obs, reward, terminated, truncated, info = env.step(action)

    scene = env.unwrapped.scene
    contacts = scene.get_contacts()

    print(f"contacts type: {type(contacts)}")
    print(f"num contacts: {len(contacts)}")
    if contacts:
        first = contacts[0]
        print(f"first contact type: {type(first)}")
        for name in dir(first):
            if name.startswith("_") or name == "cast":
                continue
            try:
                print(f"  {name}: {getattr(first, name)}")
            except Exception as exc:
                print(f"  {name}: <error reading attribute: {exc}>")

        def resolve_body_name(body):
            try:
                entity = body.entity
                return getattr(entity, "name", "<unnamed>")
            except Exception:
                return "<unknown>"

        def resolve_shape_name(shape):
            try:
                actor = shape.actor
                entity = actor.entity
                return getattr(entity, "name", "<unnamed>")
            except Exception:
                return "<unknown>"

        print("body names:")
        for body in getattr(first, "bodies", []):
            print(f"  - {resolve_body_name(body)} ({body})")

        print("shape parent names:")
        for shape in getattr(first, "shapes", []):
            print(f"  - {resolve_shape_name(shape)} ({shape})")

    env.close()


if __name__ == "__main__":
    main()
