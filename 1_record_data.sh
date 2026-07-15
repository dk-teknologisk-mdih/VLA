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
#   ./record_data.sh --name <recording_name> [--start-seq <N>]
#   ./record_data.sh --name pick_cup
#   ./record_data.sh --name pick_cup --start-seq 4   # continue from sequence 4
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
START_SEQ=1
KILL_WINDOWS=true

usage() {
    cat <<EOF
Usage: $(basename "$0") --name <recording_name> [--start-seq <N>] [--no-kill-windows]

Starts the local recording pipeline and records multiple sequences back-to-back.
Sequences are saved as <name>_N, <name>_N+1, … under vla_data/pickle/.

  1. Reset robot joints   (deoxys_control/deoxys/examples/reset_robot_joints.py)
  2. Cameras              (openteach/robot_camera.py)  — stays up the whole time
  3. Teleoperation        (openteach/teleop.py    robot=franka record=<name>_N)
  4. Data collection      (openteach/data_collect.py robot=franka demo_num=<name>_N)

While recording:
  Press 's'   — save current sequence and prepare the next one
               (new processes launch paused; press button B on Quest to begin)
  Press Enter — stop all recording and exit

Options:
  --name <name>       Base name for all sequences (required)
  --start-seq <N>     Starting sequence number (default: 1)
  --no-kill-windows   Do not close child terminator windows after recording
  -h, --help          Show this help and exit
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
        --start-seq)
            [[ $# -ge 2 ]] || { echo "error: --start-seq requires a value" >&2; exit 1; }
            START_SEQ="$2"
            [[ "$START_SEQ" =~ ^[1-9][0-9]*$ ]] || { echo "error: --start-seq must be a positive integer" >&2; exit 1; }
            shift 2
            ;;
        --start-seq=*)
            START_SEQ="${1#*=}"
            [[ "$START_SEQ" =~ ^[1-9][0-9]*$ ]] || { echo "error: --start-seq must be a positive integer" >&2; exit 1; }
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
# Close only the recording windows for the current sequence (teleop +
# data-collect), leaving the camera window running.
# --------------------------------------------------------------------------- #
close_recording_windows() {
    echo "Closing recording windows for sequence ${SEQ}..."
    local pid
    for pid in "${RECORDING_WINDOW_PIDS[@]}"; do
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
}

# --------------------------------------------------------------------------- #
# Remove a sequence folder if recording was never actually started (i.e.
# button B was never pressed, so arm_control() was never called and the
# deoxys_obs_cmd_history pickle is an empty dict — a few bytes at most).
#   $1 = sequence name (e.g. pick_cup_2)
# --------------------------------------------------------------------------- #
cleanup_empty_sequence() {
    local name="$1"
    local seq_dir="${SCRIPT_DIR}/vla_data/pickle/${name}"
    local pkl_file="${seq_dir}/deoxys_obs_cmd_history_${name}.pkl"

    [[ -d "$seq_dir" ]] || return 0

    # If the pkl file hasn't appeared yet (process still shutting down), wait.
    if [[ ! -f "$pkl_file" ]]; then
        sleep 2
    fi

    local is_empty=false
    if [[ ! -f "$pkl_file" ]]; then
        is_empty=true
    else
        local pkl_size
        pkl_size="$(stat -c%s "$pkl_file" 2>/dev/null || echo 0)"
        # An empty dict pickle is ~5 bytes; anything under 100 bytes means no
        # data was recorded (arm_control() was never called).
        if [[ "$pkl_size" -lt 100 ]]; then
            is_empty=true
        fi
    fi

    if [[ "$is_empty" == true ]]; then
        echo "Sequence '${name}' was never started (button B not pressed) — removing empty folder."
        rm -rf "$seq_dir"
    fi
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

# --------------------------------------------------------------------------- #
# Release ZMQ ports that may be held by a previous (crashed) run before
# starting any new FrankaInterface instances.  FrankaInterface binds to the
# PC-side publisher ports (SUB_PORT=5555, GRIPPER_SUB_PORT=5557) in its
# constructor, so any leftover process holding those ports must be gone first.
# --------------------------------------------------------------------------- #
for port in 5555 5557; do
    pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        echo "Releasing port $port (PIDs: $pids)..."
        echo "$pids" | xargs -r kill -TERM 2>/dev/null || true
        sleep 1
        # Force-kill anything that didn't exit cleanly.
        remaining="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
        if [[ -n "$remaining" ]]; then
            echo "$remaining" | xargs -r kill -KILL 2>/dev/null || true
        fi
    fi
done

# The robot only accepts commands from one process at a time, so run the two
# reset scripts sequentially and block until each completes before continuing.

# 1. Reset the robot arm to its home joint configuration.
run_blocking "reset-joints" "$DEOXYS_EXAMPLES_DIR" "python reset_robot_joints.py"

# # 2. Reset (open) the gripper.
# run_blocking "reset-gripper" "$OT_DIR" "python reset_gripper.py"

# 3. Start the cameras (needs a few seconds to reset/enumerate the RealSense devices).
launch "cameras" "$OT_DIR" "python robot_camera.py"
sleep 5

# Stop the recording scripts on Ctrl+C / termination of THIS script too.
trap 'stop_recording' INT TERM

# --------------------------------------------------------------------------- #
# Multi-sequence recording loop.
#
# Each iteration launches teleop + data-collect under <name>_N and waits for
# the user to interact:
#   's'   — save this sequence (SIGINT) and prepare the next one; new processes
#            launch immediately but stay PAUSED until button B is pressed on
#            the Quest, giving time to reset the scene between recordings.
#   Enter — stop all recording and exit the script.
# --------------------------------------------------------------------------- #
SEQ=$START_SEQ
RECORDING_WINDOW_PIDS=()

while true; do
    CURRENT_NAME="${NAME}_${SEQ}"
    STOP_ANNOUNCED=false

    # Start teleoperation for this sequence.
    launch "teleop-${SEQ}" "$OT_DIR" \
        "python teleop.py robot=franka record=${CURRENT_NAME}" "$TELEOP_PIDFILE"
    RECORDING_WINDOW_PIDS+=("${WINDOW_PIDS[-1]}")
    sleep 3

    # Start data collection for this sequence.
    launch "data-collect-${SEQ}" "$OT_DIR" \
        "python data_collect.py robot=franka demo_num=${CURRENT_NAME}" "$COLLECT_PIDFILE"
    RECORDING_WINDOW_PIDS+=("${WINDOW_PIDS[-1]}")

    cat <<EOF

Sequence ${SEQ} ready — recording name: '${CURRENT_NAME}'
  -> vla_data/pickle/${CURRENT_NAME}/

  Processes are PAUSED — press button B on the Quest to start recording.

  Press 's'   to save this sequence and prepare the next one.
  Press Enter to stop all recording and exit.
EOF

    announce "Recording ${SEQ} ready. Press button B to start."

    # Poll for a single keypress: 's' = save + next, Enter = stop.
    user_action="stop"
    while true; do
        if IFS= read -r -n 1 -s -t 0.5 key 2>/dev/null; then
            case "$key" in
                s|S)
                    user_action="save"
                    break
                    ;;
                $'\n'|"")
                    user_action="stop"
                    break
                    ;;
            esac
        fi
    done

    echo
    stop_recording

    if [[ "$KILL_WINDOWS" == true ]]; then
        # Give processes time to flush .pkl / video files after SIGINT.
        sleep 3
        close_recording_windows
    fi

    # Remove the folder if recording was never started (button B not pressed).
    cleanup_empty_sequence "$CURRENT_NAME"

    if [[ "$user_action" == "stop" ]]; then
        break
    fi

    # Prepare for the next sequence.
    SEQ=$((SEQ + 1))
    RECORDING_WINDOW_PIDS=()

    echo
    echo "Sequence $((SEQ - 1)) saved to vla_data/pickle/${NAME}_$((SEQ - 1))/"
    echo "Reset the scene, then press button B on the Quest to start sequence ${SEQ}."
    announce "Saved. Reset the scene, then press button B to start recording ${SEQ}."
done

if [[ "$KILL_WINDOWS" == true ]]; then
    kill_windows
fi

trap - INT TERM
rm -rf "$PID_DIR"

if [[ "$KILL_WINDOWS" == true ]]; then
    echo "All recording windows closed."
else
    echo "Recording scripts stopped. Check their windows for the saved files."
fi
