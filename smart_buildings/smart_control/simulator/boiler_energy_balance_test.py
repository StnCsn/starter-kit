"""Tests the energy-balanced hot-water tank.

Before, the tank temperature moved toward the setpoint at a fixed rate whatever
the coils drew, while gas was billed at the setpoint. A policy could charge the
tank, drop the setpoint below the return temperature and keep drawing heat at
the tank temperature for hours with the flow term at zero: billed gas fell to
0.38 of the heat the coils delivered. Now every joule the coils take is burned,
in the same step or earlier to charge the tank.
"""

import inspect

from absl.testing import absltest

from smart_buildings.smart_control.simulator import (
    boiler,
    simulator,
    simulator_flexible_floor_plan,
)
from smart_buildings.smart_control.utils import constants

STEP_S = 300.0
CAPACITY = constants.WATER_HEAT_CAPACITY * constants.WATER_DENSITY * 1.5  # J/K


def _boiler(setpoint):
  return boiler.Boiler(
      supply_water_temperature_setpoint=setpoint,
      heating_rate=0.5,
      cooling_rate=0.1,
  )


class BoilerEnergyBalanceTest(absltest.TestCase):

  def test_energy_is_conserved_over_a_charge_discharge_cycle(self):
    b = _boiler(322.0)
    t0 = b.supply_water_temperature_sensor
    burned = drawn = lost = 0.0
    for k in range(200):
      b.supply_water_temperature_setpoint = 348.0 if k % 60 < 12 else 310.0
      loss = b.compute_thermal_dissipation_rate(
          b.supply_water_temperature_sensor, 280.0
      )
      b.apply_heat_draw(50_000.0, 280.0, STEP_S)
      burned += b.compute_thermal_energy_rate(0.0, 280.0, 0.0) * STEP_S
      drawn += 50_000.0 * STEP_S
      lost += loss * STEP_S
    stored = CAPACITY * (b.supply_water_temperature_sensor - t0)
    self.assertAlmostEqual(burned, drawn + lost + stored, delta=1e-6 * burned)

  def test_low_setpoint_does_not_give_free_heat(self):
    b = _boiler(348.0)
    b.supply_water_temperature_setpoint = 310.0
    t0 = b.supply_water_temperature_sensor
    b.apply_heat_draw(50_000.0, 280.0, STEP_S)
    self.assertEqual(b.compute_thermal_energy_rate(0.0, 280.0, 0.0), 0.0)
    # the tank pays for the draw: 50 kW for 5 min out of 6.3 MJ/K
    self.assertLess(b.supply_water_temperature_sensor, t0 - 2.3)

  def test_both_simulators_advance_the_tank(self):
    # With energy_balance on, the boiler no longer moves its own temperature on
    # a sensor read, so a simulator that skipped apply_heat_draw would freeze
    # the tank and bill no gas at all.
    for step in (
        simulator.Simulator.execute_step_sim,
        simulator_flexible_floor_plan.SimulatorFlexibleGeometries.execute_step_sim,
    ):
      self.assertIn(
          "hot_water_system.apply_heat_draw(", inspect.getsource(step)
      )


if __name__ == "__main__":
  absltest.main()
