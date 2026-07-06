#!/usr/bin/env bash
#
# record_data.sh — Start all LOCAL robot-data-recording processes with one command.
#
# The NUC-side processes (franka arm + gripper controllers launched via
# auto_arm.sh / auto_gripper.sh) are assumed to already be running. This script
# only starts the pieces that live on THIS machine, filling in the recording
# name automatically so nothing has to be typed by hand.
#
# Usage:
#   ./record_data.sh --name <recording_name>
#   ./record_data.sh --name pick_cup
#
# The single <recording_name> is used for both:
#   * teleop.py       -> record=<name>      (deoxys_obs_cmd_history_<name>.pkl)
#   * data_collect.py -> demo_num=<name>    (vla_data/pickle/<name>/)
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
NAME=""
KILL_WINDOWS=true

usage() {
    cat <<EOF
Usage: $(basename "$0") --name <recording_name> [--no-kill-windows]

Starts the local recording pipeline (each component in its own terminator window):
  1. Reset robot joints   (deoxys_control/deoxys/examples/reset_robot_joints.py)
  2. Reset gripper        (openteach/reset_gripper.py)
  3. Cameras              (openteach/robot_camera.py)
  4. Teleoperation        (openteach/teleop.py    robot=franka record=<name>)
  5. Data collection      (openteach/data_collect.py robot=franka demo_num=<name>)

Options:
  --name <name>    Name used for both record= and demo_num= (required)
  --no-kill-windows   Do not close child terminator windows once recording finishes
  -h, --help       Show this help and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            [[ $# -ge 2 ]] || { echo "error: --name requires a value" >&2; exit 1; }
            NAME="$2"
            shift 2
            ;;
        --name=*)
            NAME="${1#*=}"
            shift
            ;;
        --no-kill-windows)
            KILL_WINDOWS=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$NAME" ]]; then
    echo "error: --name is required" >&2
    usage >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Paths / configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OT_DIR="$SCRIPT_DIR/openteach"
DEOXYS_EXAMPLES_DIR="$SCRIPT_DIR/deoxys_control/deoxys/examples"
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"

# Temp dir used to share the recording scripts' process-group ids back to this
# script so they can be stopped from here.
PID_DIR="$(mktemp -d)"
TELEOP_PIDFILE="$PID_DIR/teleop.pgid"
COLLECT_PIDFILE="$PID_DIR/data_collect.pgid"

# PIDs of every terminator window launched by this script, so they can be
# closed together when --kill-windows is requested.
WINDOW_PIDS=()

for d in "$OT_DIR" "$DEOXYS_EXAMPLES_DIR"; do
    if [[ ! -d "$d" ]]; then
        echo "error: expected directory not found: $d" >&2
        exit 1
    fi
done

if ! command -v terminator >/dev/null 2>&1; then
    echo "error: 'terminator' is not installed or not on PATH." >&2
    echo "       Install it (e.g. 'sudo apt install terminator') or adapt this script" >&2
    echo "       to your terminal emulator of choice." >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Helper: launch a command in its own terminator window.
#   $1 = window title
#   $2 = working directory
#   $3 = command to run (inside the .venv environment)
#   $4 = (optional) pidfile. When given, the command is run as a background job
#        (so bash job-control puts it in its own process group) and its PID —
#        which equals its process-group id — is written to the file so this
#        script can later signal the whole group.
# The window stays open after the command exits so output can be inspected.
# --------------------------------------------------------------------------- #
launch() {
    local title="$1"
    local workdir="$2"
    local cmd="$3"
    local pidfile="${4:-}"

    # --no-dbus (-u) makes every terminator window its own independent process
    # instead of handing off to a single-instance master over DBus. That way the
    # PID captured in $! below really is the window's process and kill_windows()
    # can close each window with a signal.
    #
    # Each independent (--no-dbus) instance tries to grab the same global
    # hide_window hotkey and prints a harmless GTK "Binding ... failed" warning
    # to this parent terminal, so terminator's stderr is discarded.
    if [[ -n "$pidfile" ]]; then
        terminator -u -T "$title" -e \
            "bash -ic 'cd \"$workdir\" && source $VENV_PATH && echo \"[$title] \$ $cmd\"; $cmd & echo \$! > \"$pidfile\"; wait \$!; echo; echo \"[$title] process exited — press Enter to close\"; read; exec bash'" 2>/dev/null &
    else
        terminator -u -T "$title" -e \
            "bash -ic 'cd \"$workdir\" && source $VENV_PATH && echo \"[$title] \$ $cmd\" && $cmd; echo; echo \"[$title] process exited — press Enter to close\"; read; exec bash'" 2>/dev/null &
    fi
    # Remember this window's PID so it can be closed later if requested.
    WINDOW_PIDS+=("$!")
}

# --------------------------------------------------------------------------- #
# Close every child terminator window launched by this script.
# --------------------------------------------------------------------------- #
kill_windows() {
    echo
    echo "Closing child terminator windows..."
    local pid
    for pid in "${WINDOW_PIDS[@]}"; do
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
}

# --------------------------------------------------------------------------- #
# Helper: run a command to completion in THIS shell (blocking).
# Used for the robot reset scripts: only one process may command the robot at a
# time, so these must finish before anything else (teleop) starts talking to it.
#   $1 = label (for logging)
#   $2 = working directory
#   $3 = command to run (inside the .venv environment)
# --------------------------------------------------------------------------- #
run_blocking() {
    local label="$1"
    local workdir="$2"
    local cmd="$3"

    echo "[$label] $ $cmd"
    ( cd "$workdir" && source "$VENV_PATH" && eval "$cmd" )
    echo "[$label] done"
}

# --------------------------------------------------------------------------- #
# Speak a short message out loud through the Logitech MeetUp speakerphone.
# Best effort only: uses speech-dispatcher (spd-say) and pins playback to the
# MeetUp PipeWire sink when it can be located, otherwise falls back to the
# default output device. Never fails the pipeline if audio is unavailable.
#   $1 = text to speak
# --------------------------------------------------------------------------- #
announce() {
    local text="$1"

    command -v spd-say >/dev/null 2>&1 || return 0

    # Try to locate the MeetUp output sink so the announcement plays on it
    # regardless of which device happens to be the system default.
    local node_name=""
    if command -v wpctl >/dev/null 2>&1; then
        local sink_id
        sink_id="$(wpctl status 2>/dev/null \
            | sed -n '/Sinks:/,/Sources:/p' \
            | grep -i 'meetup' \
            | grep -oE '[0-9]+\.' | head -1 | tr -d '.')"
        if [[ -n "$sink_id" ]]; then
            node_name="$(wpctl inspect "$sink_id" 2>/dev/null \
                | grep -oE 'node.name = "[^"]+"' \
                | sed -E 's/.*"([^"]+)".*/\1/')"
        fi
    fi

    if [[ -n "$node_name" ]]; then
        PULSE_SINK="$node_name" spd-say -w "$text" 2>/dev/null \
            || spd-say -w "$text" 2>/dev/null || true
    else
        spd-say -w "$text" 2>/dev/null || true
    fi
}

# --------------------------------------------------------------------------- #
# Stop the two recording scripts by sending SIGINT to their process groups.
# This lets the python scripts (and their multiprocessing children) run their
# KeyboardInterrupt handlers and finish writing their .pkl / video files —
# exactly as if the user had pressed Ctrl+C in each window.
# --------------------------------------------------------------------------- #
STOP_ANNOUNCED=false
stop_recording() {
    echo
    echo "Stopping recording scripts (teleop + data_collect)..."
    local pf pid
    for pf in "$TELEOP_PIDFILE" "$COLLECT_PIDFILE"; do
        [[ -f "$pf" ]] || continue
        pid="$(cat "$pf" 2>/dev/null || true)"
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            # Negative pid -> signal the whole process group; fall back to the
            # bare pid if the group signal isn't permitted.
            kill -INT -"$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
        fi
    done

    # Announce over the Logitech MeetUp that recording has stopped (once only,
    # since this may run both from the trap and the explicit call below).
    if [[ "$STOP_ANNOUNCED" != true ]]; then
        STOP_ANNOUNCED=true
        announce "Stopped recording"
    fi
}

echo "Starting local recording pipeline with name: '$NAME'"

# The robot only accepts commands from one process at a time, so run the two
# reset scripts sequentially and block until each completes before continuing.

# 1. Reset the robot arm to its home joint configuration.
run_blocking "reset-joints" "$DEOXYS_EXAMPLES_DIR" "python reset_robot_joints.py"

# # 2. Reset (open) the gripper.
# run_blocking "reset-gripper" "$OT_DIR" "python reset_gripper.py"

# 3. Start the cameras (needs a few seconds to reset/enumerate the RealSense devices).
launch "cameras" "$OT_DIR" "python robot_camera.py"
sleep 5

# 4. Start teleoperation, recording the action/observation history under <name>.
launch "teleop" "$OT_DIR" "python teleop.py robot=franka record=$NAME" "$TELEOP_PIDFILE"
sleep 3

# 5. Start data collection, saving to vla_data/pickle/<name>/.
launch "data-collect" "$OT_DIR" "python data_collect.py robot=franka demo_num=$NAME" "$COLLECT_PIDFILE"

cat <<EOF

All local components launched (each in its own terminator window):
  cameras, teleop, data-collect
  (reset-joints and reset-gripper already ran and completed)

Recording name : $NAME
  teleop record= $NAME
  demo_num=       $NAME  -> vla_data/pickle/$NAME/

The cameras / reset windows keep running independently.
EOF

# Stop the recording scripts on Ctrl+C / termination of THIS script too.
trap 'stop_recording' INT TERM

# Everything is up — announce over the Logitech MeetUp that recording can begin.
announce "Start recording"

echo
read -r -p "Press Enter here to STOP recording (teleop + data_collect)... " _

stop_recording

if [[ "$KILL_WINDOWS" == true ]]; then
    # Give the recording scripts a moment to flush their .pkl / video files
    # after receiving SIGINT before tearing their windows down.
    sleep 3
    kill_windows
fi

trap - INT TERM
rm -rf "$PID_DIR"

if [[ "$KILL_WINDOWS" == true ]]; then
    echo "Recording scripts signalled to stop and child windows closed."
else
    echo "Recording scripts signalled to stop. Check their windows for the saved files."
fi
