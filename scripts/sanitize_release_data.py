"""Redact machine-specific information from public JSON and JSONL datasets."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

# URL matches take precedence over filesystem paths to preserve public sources.
_URL = r"(?:https?|socks5h?|file)://[^\s\"'<>`\\]+"
_LOCAL = (
    r"/(?:Users|Userz|home|root|blue|orange|scratch|mnt|lustre|gpfs|"
    r"private/var|Library|opt|workspace|REDACTED_HOME|REDACTED_TMP|"
    r"REDACTED_ROOT|var/(?:tmp|folders)|usr/local)/[^\s\"'<>`\\)\]},;]+"
)
_WINDOWS = r"\b[A-Za-z]:\\+(?:Users|Documents and Settings|Program Files|Projects|home)\\+[^\r\n\"'<>`]+"
_VENV = r"(?<![\w])\.venv(?:-[\w-]+)?(?:/[^\s\"'<>`\\)\]},;]+)?"
_TOKEN = (
    r"(?<![\w-])(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|"
    r"AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
)
_PRIVATE_KEY = r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z]+ )?PRIVATE KEY-----"
_MATCH = re.compile(
    f"(?P<url>{_URL})|(?P<private_key>{_PRIVATE_KEY})|(?P<credential>{_TOKEN})|"
    f"(?P<local_path>{_LOCAL})|(?P<windows_path>{_WINDOWS})|(?P<environment>{_VENV})|"
    r"(?P<gpu>\bGPU-[a-fA-F0-9-]{16,})"
)
_NETWORK = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:[a-z_]*api[_-]?key|access_token|auth_token|client_secret|"
    r"authorization|https?_proxy|all_proxy|base_url|api_base)\b[\\\"'\s]*[:=]\s*\\*[\"'])"
    r"([^\s\"'\\,;}\]]+)"
)
_HEADERS = re.compile(r"(?i)(\b(?:set-cookie|cookie|authorization)\s*:\s*)([^\\\r\n\"']+)")


def sanitize_headers(value: str, counts: Counter) -> str:
    def replace(match: re.Match) -> str:
        if match.group(2).strip() == _REDACTED:
            return match.group()
        counts['authentication_header'] += 1
        return match.group(1) + _REDACTED
    return _HEADERS.sub(replace, value)


_OWNERS = re.compile(
    r"([bcdlps-][rwxstST-]{9}[+@.]?[ \t]+\d+[ \t]+)"
    r"([^\s\\]+)[ \t]+([^\s\\]+)(?=[ \t]+\d+[ \t]+[A-Z][a-z]{2}[ \t]+\d)"
)


def sanitize_owners(value: str, counts: Counter) -> str:
    def replace(match: re.Match) -> str:
        if match.group(2) == '[REDACTED_USER]' and match.group(3) == '[REDACTED_GROUP]':
            return match.group()
        counts['filesystem_ownership'] += 1
        return match.group(1) + '[REDACTED_USER] [REDACTED_GROUP]'
    return _OWNERS.sub(replace, value)


_SENSITIVE_KEYS = frozenset({
    'api_key', 'apikey', 'access_token', 'auth_token', 'client_secret',
    'password', 'authorization', 'base_url', 'api_base', 'endpoint',
    'http_proxy', 'https_proxy', 'all_proxy', 'hostname', 'username',
    'cwd', 'working_dir', 'python_executable', 'cuda_visible_devices',
    'hf_home', 'conda_prefix', 'virtual_env',
})
_REDACTED = '[REDACTED]'


def sanitize_text(value: str, counts: Counter) -> str:
    def replace(match: re.Match) -> str:
        kind = match.lastgroup
        text = match.group()
        if kind == 'url':
            try:
                parsed = urlsplit(text.rstrip('.,;:)'))
                host = (parsed.hostname or '').lower()
            except ValueError:
                counts['endpoint'] += 1
                return 'https://api.example.invalid'
            if host.endswith('.invalid') and not parsed.username and not parsed.query:
                return text
            try:
                ipaddress.ip_address(host)
                address = True
            except ValueError:
                address = False
            sensitive_query = re.search(r'(?i)(?:[?&])(?:key|api_key|token|access_token|signature|auth)=', text)
            endpoint = (
                parsed.scheme not in {'http', 'https'} or parsed.username is not None
                or address or host in {'localhost', 'testserver'} or host.endswith('.local')
                or any(label.startswith('api') or label.endswith('api') for label in host.split('.'))
                or '/api/' in parsed.path or parsed.path.endswith('/api')
                or bool(re.search(r'/(?:v\d+/)?(?:chat/completions|responses|messages)(?:[/?]|$)', parsed.path))
            )
            if endpoint:
                counts['endpoint'] += 1
                return 'https://api.example.invalid'
            if sensitive_query:
                counts['url_credentials'] += 1
                return text[:text.index('?')]
            return text
        counts[kind] += 1
        if kind in {'local_path', 'windows_path', 'environment'}:
            return '[REDACTED_LOCAL_PATH]'
        return _REDACTED

    if 'r--' in value or 'rwx' in value or 'rw-' in value:
        value = sanitize_owners(value, counts)
    lowered = value.lower()
    if 'cookie' in lowered or 'authorization' in lowered:
        value = sanitize_headers(value, counts)
    if any(marker in value for marker in (
        '://', '/Users/', '/Userz/', '/home/', '/root/', '/blue/', '/orange/',
        '/scratch/', '/mnt/', '/lustre/', '/gpfs/', '/private/var/', '/Library/',
        '/opt/', '/workspace/', '/REDACTED_', '/var/tmp/', '/var/folders/',
        '/usr/local/', ':\\', '.venv', 'sk-', 'hf_', 'AKIA', 'gh', 'GPU-', 'PRIVATE KEY',
    )):
        value = _MATCH.sub(replace, value)
    def network(match: re.Match) -> str:
        address = match.group()
        try:
            ip = ipaddress.IPv4Address(address)
        except ipaddress.AddressValueError:
            return address
        first = int(address.split('.')[0])
        # Four-part software versions occur frequently in benchmark source text.
        context = value[max(0, match.start() - 100):match.start()].lower()
        network_context = bool(re.search(r'ip|address|network|proxy|host', context))
        if (first in range(1, 10) and not network_context) or (address.endswith('.0.0.0') and first > 127):
            return address
        if address.endswith('.0.0.0') and first in range(100, 127):
            return address
        counts['network_address'] += 1
        return '[REDACTED_IP]'

    if value.count('.') >= 3:
        value = _NETWORK.sub(network, value)

    def assignment(match: re.Match) -> str:
        current = match.group(2)
        if current.startswith(('[REDACTED', 'REDACTED', 'https://api.example.invalid')):
            return match.group()
        # Keep references to environment variables and absent values intact.
        if current in {'None', 'null', 'True', 'False'} or current.startswith(('os.', 'getenv(', '${')):
            return match.group()
        counts['configuration_value'] += 1
        return match.group(1) + _REDACTED

    if any(marker in lowered for marker in ('api', 'token', 'secret', 'authorization', 'proxy', 'base_url')):
        value = _ASSIGNMENT.sub(assignment, value)
    return value


def sanitize(value, counts: Counter):
    if isinstance(value, str):
        return sanitize_text(value, counts)
    if isinstance(value, list):
        return [sanitize(item, counts) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            new_key = sanitize_text(key, counts)
            if new_key in result:
                raise ValueError('Redaction would merge distinct object keys')
            if key.lower() in _SENSITIVE_KEYS and item not in (None, '', _REDACTED):
                counts['configuration_field'] += 1
                result[new_key] = _REDACTED
            else:
                result[new_key] = sanitize(item, counts)
        return result
    return value


def sanitize_file(path: Path) -> dict:
    counts = Counter()
    rows = changed = 0
    before = hashlib.sha256()
    after = hashlib.sha256()
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as target, path.open('rb') as source:
            chunks = source if path.suffix == '.jsonl' else [source.read()]
            for raw in chunks:
                before.update(raw)
                if not raw.strip():
                    target.write(raw)
                    after.update(raw)
                    continue
                original = json.loads(raw)
                clean = sanitize(original, counts)
                rows += 1
                if clean != original:
                    changed += 1
                    output = (json.dumps(clean, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
                else:
                    output = raw
                target.write(output)
                after.update(output)
        if changed:
            os.chmod(name, path.stat().st_mode & 0o777)
            os.replace(name, path)
        else:
            os.unlink(name)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return {'rows': rows, 'changed_rows': changed,
            'redactions': {k: v for k, v in sorted(counts.items()) if v},
            'original_sha256': before.hexdigest(), 'sha256': after.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    paths = sorted(p for p in args.data_dir.rglob('*')
                   if p.suffix in {'.json', '.jsonl'} and p.resolve() != args.report.resolve())
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(sanitize_file, path): path for path in paths}
        for job in as_completed(jobs):
            path = jobs[job]
            result = job.result()
            results[str(path.relative_to(args.data_dir))] = result
            print(json.dumps({'file': str(path), **result}), flush=True)
    # Synchronize content hashes in manifests after rewriting their data files.
    for path in paths:
        if path.name != 'manifest.json':
            continue
        value = json.loads(path.read_text())
        siblings = sorted(path.parent.glob('*.jsonl'))
        if not siblings:
            continue
        value['sha256'] = {p.name: results[str(p.relative_to(args.data_dir))]['sha256'] for p in siblings}
        value['sanitized'] = True
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        results[str(path.relative_to(args.data_dir))]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'schema_version': 'aria-release-redaction-v1', 'files': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
