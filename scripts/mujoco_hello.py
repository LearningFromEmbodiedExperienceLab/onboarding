"""Open this file in the editor after generating MuJoCo stubs.

Hover ``MjModel``, ``MjData``, and ``mj_step``. Without stubs they are
``Unknown``; with stubs the editor shows real signatures.
"""

import mujoco as mj

XML = """
<mujoco>
  <worldbody>
    <geom type="sphere" size="0.1"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    model = mj.MjModel.from_xml_string(XML)
    data = mj.MjData(model)
    mj.mj_step(model, data)
    print("qpos", data.qpos)


if __name__ == "__main__":
    main()
