from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.core.env import env_bool, env_int, env_str, load_env_file


class EnvTest(TestCase):
    def test_load_env_file_sets_missing_environment_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".env"
            path.write_text(
                """
# comment
MT5_LOGIN=10344
MT5_PASSWORD="#secret"
export MT5_PORTABLE=true
MT5_TIMEOUT_MS=45000
""",
                encoding="utf-8",
            )
            old_values = {
                key: os.environ.get(key)
                for key in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_PORTABLE", "MT5_TIMEOUT_MS")
            }
            try:
                for key in old_values:
                    os.environ.pop(key, None)

                loaded = load_env_file(path)

                self.assertEqual(loaded["MT5_LOGIN"], "10344")
                self.assertEqual(env_str("MT5_PASSWORD"), "#secret")
                self.assertTrue(env_bool("MT5_PORTABLE"))
                self.assertEqual(env_int("MT5_TIMEOUT_MS"), 45_000)
            finally:
                for key, value in old_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_load_env_file_rejects_non_key_value_lines(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".env"
            path.write_text("not valid\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
                load_env_file(path)
