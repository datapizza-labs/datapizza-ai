import pytest

from datapizza.tools.filesystem import string_matches_patterns

test_cases = [
    # (file_path, regex_patterns, expected_result)
    ("/home/user/file.sys", ["*.sys"], True),
    ("/home/user/file.sys", ["/home/*.sys"], True),
    ("/home/user/file.sys", ["/home/user/*.sys"], True),
    ("/home/user/file.sys", ["*.txt"], False),
    # Additional test cases
    ("/var/log/syslog.log", ["*.log"], True),
    ("/var/log/syslog.log", ["syslog.log"], False),
    ("/var/log/syslog.log", ["*syslog.log"], True),
    ("/var/log/syslog.log", ["/var/log/syslog.log"], True),
    ("/var/log/syslog.log", ["*.txt"], False),
    ("/data/report.pdf", ["*.PDF"], True),  # Case-insensitive match
    ("/data/report.pdf", ["report.*"], False),
    ("/data/report.pdsss", ["*report.*"], True),
    ("/data/report/pdsss", ["*report*"], True),
    ("/data/archive.tar.gz", ["*.tar.gz"], True),
    ("/data/archive.tar.gz", ["*.tar.gz", "*.txt"], True),
    ("/data/archive.tar.gz", ["*.zip", "*.txt"], False),
    ("/data/archive.tar.gz", ["*.zip", "/data/archive.tar.gz"], True),
    ("/data/archive.tar.gz", ["*"], True),
    # Complex regex patterns with explanations
    # 1. Match any .log file in /var/log directory
    ("/var/log/syslog.log", [r"^/var/log/.*\.log$"], True),
    # 2. Match .txt files in /var/log with specific naming
    ("/var/log/syslog.log", [r"^/var/log/[^/]+\.txt$"], False),
    # 3. Match date-formatted PDF reports (YYYYMMDD)
    ("/data/reports/20230801.pdf", [r"^/data/reports/\d{8}\.pdf$"], True),
    # 4. Reject non-date formatted PDF reports
    ("/data/reports/summary.pdf", [r"^/data/reports/\d{8}\.pdf$"], False),
    # 5. Match backup files in daily/weekly subdirectories
    ("/backup/daily/logs.tar.gz", [r"^/backup/(?:daily|weekly)/.*"], True),
    # 6. Reject monthly backup files (not in daily/weekly)
    ("/backup/monthly/logs.tar.gz", [r"^/backup/(?:daily|weekly)/.*"], False),
    # 7. Match localized document files (en/it)
    ("/home/user/docs/resume_en.docx", [r"^/home/user/docs/\w+_(en|it)\.docx$"], True),
    # 8. Reject non-supported language documents
    ("/home/user/docs/resume_fr.docx", [r"^/home/user/docs/\w+_(en|it)\.docx$"], False),
]


@pytest.mark.parametrize("file_path, patterns, expected", test_cases)
def test_matches_regex(file_path, patterns, expected):
    result = string_matches_patterns(file_path, patterns)
    assert result == expected, f"Failed for {file_path} with patterns {patterns}"


def test_empty_patterns_list():
    assert (
        string_matches_patterns("/any/path", []) == True
    ), "Should return True with empty patterns list"
