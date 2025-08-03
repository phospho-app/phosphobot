# Feetech STS3215 utility scripts

## Installation

1. [Clone or download this git repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

```bash
git clone git@github.com:phospho-app/phosphobot.git
```

2. Install [uv.](https://docs.astral.sh/uv/)

MacOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:

```pwh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installing uv, restart your terminal.

3. You can now run the script `filename.py` with `un run filename.py`

## Initializing a Feetech STS3215 servomotor

This procedure is useful to replace a broken servomotor with a new one.

1. Connect the servo to the waveshare servo bus. Connect the servo bus to the power and to your computer using USB C.

2. Find out the motor bus of your waveshare servo bus.

```
cd scripts/feetech
uv run find_motor_bus.py
```

You can also use `phosphobot info` to do that.

3. Then, to initialize a servo with id `1` (base motor), you do :

```bash
uv run configure_motor.py \
  --port /dev/tty.usbmodem58FD0162361 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```

After running the command, you should hear a spinning sound and the motor should go

**Note:** Change the **port** depending on the value of `find_motor_bus.py` and and the **id** depending on what servo you're programming!

On the SO-100 and SO-101 robot, the ids of the motors start at 1 (base) and go up to 6 (gripper).

**Note**: On windows, you need to remove the breaklines and remove the backlashes:

```pwh
uv run configure_motor.py --port COM4 --brand feetech --model sts3215 --baudrate 1000000 --ID 1
```

4. After programming your servo and replacing it, you then need to **recalibrate** your robot arm. ([Example with SO-100 and phosphobot](https://www.youtube.com/watch?v=65DW8yLcRmM))

## Troubleshooting communication issues with Feetech STS3215 servomotors

Here are some common error messages when dealing with Feetech.

```
Error reading torque status: Read failed due to communication error on port /dev/cu.usbmodel5A460913131 for group_key Torque_Enable_shoulder_pan_shoulder_lift_elbow_flex_wrist_flex_wrist_roll_gripper: [TxRxResult] Incorrect status packet!
```

```
Error writing motor position: (6, 'Device not configured')
```

Frequent solutions:

### Fix: Connection check

Make sure servos cables are properly plugged in and there are no lose wires, especially if you built the device yourself.

Make sure the Waveshare servobus is plugged in to a power source.

### Fix: Firmware check: all servos need to have the same Feetech firmware version

The communnication issue may come from the fact that one STS3215 servo doesn't have the same **firmware version** than the other ones. For a robot like the SO-100 or SO-101 to work, **all servos need to have the same firmware version**.

To change the firmware version of a servo, you need a Windows computer. Then, download the [Feetech software **FD1.9.8.3**](https://www.feetechrc.com/software.html) (windows-only) by clicking the 点击下载 button.

Then, refer to this [user manual](https://www.feetechrc.com/Data/feetechrc/upload/file/20201127/start%20%20tutorial201015.pdf) on how to update the firmware of a servo.

## Other models

These scripts were also tested [with STS3250](https://www.youtube.com/watch?v=SCX7PV1M9-k) and worked well.
