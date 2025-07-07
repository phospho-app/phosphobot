"""
This function summarizes a test from a log file.
Output is a markdown file.
"""

import sys


def main(log_file: str, out_file: str):
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Detect important lines

    # Summary line: the last of the log file
    tests_summary = "\n".join(lines[-2:])  # Last line is empty
    # If there is "failed" in the summary line, it means that there are failed tests.
    # Fetch every from the ==== summary === to the end
    some_failed = "failed" in tests_summary or "error" in tests_summary
    if some_failed:
        summary_index = 2
        for i, line in enumerate(reversed(lines)):
            if "== FAILURES ==" in line:
                summary_index = i + 2
                break
        tests_summary = "".join(lines[-summary_index:])

    # WebSocket performance lines
    performance_line_30Hz: str = ""
    performance_line_500Hz: str = ""
    performance_recording_line_500Hz: str = ""

    # UDP performance lines
    udp_performance_line_30Hz: str = ""
    udp_performance_line_500Hz: str = ""
    udp_performance_line_1000Hz: str = ""
    udp_performance_recording_line_500Hz: str = ""

    # Parse the log file
    for line in lines:
        # WebSocket performance
        if "[TEST_PERFORMANCE_30Hz]" in line:
            performance_line_30Hz = line[line.find("[TEST_PERFORMANCE_30Hz]") :]
        if "[TEST_PERFORMANCE_500Hz]" in line:
            performance_line_500Hz = line[line.find("[TEST_PERFORMANCE_500Hz]") :]
        if "[TEST_RECORDING_PERFORMANCE_500Hz]" in line:
            performance_recording_line_500Hz = line[
                line.find("[TEST_RECORDING_PERFORMANCE_500Hz]") :
            ]

        # UDP performance
        if "[TEST_UDP_PERFORMANCE_30Hz]" in line:
            udp_performance_line_30Hz = line[line.find("[TEST_UDP_PERFORMANCE_30Hz]") :]
        if "[TEST_UDP_PERFORMANCE_500Hz]" in line:
            udp_performance_line_500Hz = line[
                line.find("[TEST_UDP_PERFORMANCE_500Hz]") :
            ]
        if "[TEST_UDP_PERFORMANCE_1000Hz]" in line:
            udp_performance_line_1000Hz = line[
                line.find("[TEST_UDP_PERFORMANCE_1000Hz]") :
            ]
        if "[TEST_UDP_RECORDING_PERFORMANCE_500Hz]" in line:
            udp_performance_recording_line_500Hz = line[
                line.find("[TEST_UDP_RECORDING_PERFORMANCE_500Hz]") :
            ]

    # Build a simple markdown summary
    emoji = "✅" if not some_failed else "❌"
    summary = []
    summary.append(f"## {emoji} API integrations tests")

    # WebSocket Performance Section
    summary.append("### WebSocket Performance")
    if performance_line_30Hz:
        summary.append(performance_line_30Hz.strip())
    if performance_line_500Hz:
        summary.append(performance_line_500Hz.strip())
    if performance_recording_line_500Hz:
        summary.append(performance_recording_line_500Hz.strip())

    # UDP Performance Section
    summary.append("### UDP Performance")
    if udp_performance_line_30Hz:
        summary.append(udp_performance_line_30Hz.strip())
    if udp_performance_line_500Hz:
        summary.append(udp_performance_line_500Hz.strip())
    if udp_performance_line_1000Hz:
        summary.append(udp_performance_line_1000Hz.strip())
    if udp_performance_recording_line_500Hz:
        summary.append(udp_performance_recording_line_500Hz.strip())

    summary.append("### Pytests logs summary")

    if len(tests_summary) >= 500:
        # show only the last 500 characters
        tests_summary = tests_summary[-500:]
    summary.append(f"```{tests_summary}```")

    if not some_failed:
        summary.append(":tada: **All tests passed!** ")
    else:
        summary.append(
            "❌ **Some tests failed**. Check the logs in Github Actions for details."
        )

    # Write out the summary
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python summarize_test_results.py <log_file> <out_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
