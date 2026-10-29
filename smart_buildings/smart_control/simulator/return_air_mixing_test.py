"""The return air an air handler recirculates is the airflow-weighted mixture of its VAVs' extract.

With unequal airflow per unit floor area, an area-weighted average is not a mixture of the streams that left the
zones: it can be warmer than all of them (heat from nothing) or cooler (cooling from nothing).
"""

from pathlib import Path

import numpy as np
from absl.testing import absltest

from smart_buildings.evaluate import load_hybrid_environment

_SCENARIO_ROOT = Path('smart_buildings/training_scenarios')


def _first_scenario() -> Path:
  building = sorted(p for p in _SCENARIO_ROOT.iterdir() if p.is_dir())[0]
  strip = sorted(p for p in building.iterdir() if p.is_dir())[0]
  return strip / 'config.gin'


class ReturnAirMixingTest(absltest.TestCase):

  def test_recirculated_air_is_the_airflow_weighted_mixture(self):
    env = load_hybrid_environment(_first_scenario())
    env.reset()
    sim = env.building.simulator
    hvac = sim.hvac
    rooms = sim.building.room_dict

    # An air handler with at least two VAVs; a cold VAV at full airflow and every other VAV at its minimum.
    by_ahu = {}
    for v_id, vav in hvac.vavs.items():
      if hvac.get_zones_for_vav(v_id):
        by_ahu.setdefault(vav.air_handler.device_id(), []).append(v_id)
    ahu_id, v_ids = next((a, v) for a, v in by_ahu.items() if len(v) >= 2)
    cold = v_ids[0]
    for v_id in v_ids:
      vav = hvac.vavs[v_id]
      vav.damper_setting = 1.0 if v_id == cold else vav.min_damper_setting
      for zone in hvac.get_zones_for_vav(v_id):
        for r, c in rooms[zone]:
          sim.building.temp[r, c] = 283.15 if v_id == cold else 298.15

    avg = sim.building.get_zone_average_temps()
    flow_sum = temp_flow = area_sum = temp_area = 0.0
    for v_id in v_ids:
      zones = hvac.get_zones_for_vav(v_id)
      area = sum(len(rooms[z]) for z in zones)
      temp = sum(avg[z] * len(rooms[z]) for z in zones) / area
      flow = hvac.vavs[v_id].unrestricted_flow_rate_demand
      flow_sum, temp_flow = flow_sum + flow, temp_flow + flow * temp
      area_sum, temp_area = area_sum + area, temp_area + area * temp

    got = sim._recirculation_temps(avg)[ahu_id]  # pylint: disable=protected-access
    self.assertAlmostEqual(got, temp_flow / flow_sum, places=9)
    # Not vacuous: the old area weighting gives a clearly warmer return here.
    self.assertGreater(temp_area / area_sum - got, 0.5)
    # A mixture lies between its coldest and warmest stream.
    self.assertGreaterEqual(got, 283.15 - 1e-9)
    self.assertLessEqual(got, 298.15 + 1e-9)
    self.assertTrue(np.isfinite(got))


if __name__ == '__main__':
  absltest.main()
