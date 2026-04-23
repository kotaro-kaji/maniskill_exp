import math

def get_gripper_g2_position(pulse):
    pos = math.sin(math.radians(pulse / 18.28 - 8.33)) * 110 + 16
    return float(pos)


if __name__ == "__main__":
    print(get_gripper_g2_position(0))
    print(get_gripper_g2_position(840))
    print(get_gripper_g2_position(850))
    print(get_gripper_g2_position(860))
