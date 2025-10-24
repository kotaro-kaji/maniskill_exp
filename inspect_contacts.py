import gymnasium as gym

import mani_skill.envs  # registers built-in envs
import task_pushcube_beatiful  # registers custom env


def main():
    env = gym.make(
        "MyPushCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
        sim_backend="cpu",
    )
    env.reset(seed=0)
    scene = env.unwrapped.scene
    contacts = scene.get_contacts()

    print(f"contacts type: {type(contacts)}")
    print(f"num contacts: {len(contacts)}")

    if len(contacts) > 0:
        first = contacts[0]
        attrs = [
            attr for attr in dir(first) if not attr.startswith("_") and attr != "cast"
        ]
        print(f"first contact type: {type(first)}")
        print("public attributes/methods:")
        for name in attrs:
            try:
                value = getattr(first, name)
                print(f"  - {name}: {value}")
            except Exception as exc:
                print(f"  - {name}: <error reading attribute: {exc}>")

    env.close()


if __name__ == "__main__":
    main()
