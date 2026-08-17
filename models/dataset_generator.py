"""
dataset_generator.py

Synthesizes realistic AWS CloudTrail JSON events, including both benign user activity
and simulated malicious activity (such as unauthorized privilege escalation, unusual API calls,
credential harvesting, and defense evasion).
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd


class CloudTrailDatasetGenerator:
    """Generates synthetic AWS CloudTrail logs for threat detection research."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.benign_users = [
            {"type": "IAMUser", "arn": "arn:aws:iam::123456789012:user/alice", "userName": "alice"},
            {"type": "IAMUser", "arn": "arn:aws:iam::123456789012:user/bob", "userName": "bob"},
            {"type": "AssumedRole", "arn": "arn:aws:iam::123456789012:role/DeveloperRole", "userName": "DeveloperRole"},
            {"type": "AssumedRole", "arn": "arn:aws:iam::123456789012:role/AdminRole", "userName": "AdminRole"},
        ]
        self.malicious_users = [
            {"type": "IAMUser", "arn": "arn:aws:iam::123456789012:user/compromised_charlie", "userName": "compromised_charlie"},
            {"type": "IAMUser", "arn": "arn:aws:iam::123456789012:user/temp_contractor", "userName": "temp_contractor"},
        ]

        self.benign_ips = [
            "192.168.1.50", "192.168.1.51", "10.0.0.15", "10.0.0.22", "54.210.12.34"
        ]
        self.malicious_ips = [
            "185.220.101.5", "203.0.113.88", "198.51.100.42", "45.154.255.19"
        ]

        self.benign_user_agents = [
            "aws-cli/2.15.0 Python/3.11.4 Linux/6.5.0-generic",
            "Boto3/1.34.0 Python/3.11.4 Linux/6.5.0-generic",
            "SMCli/2.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        self.malicious_user_agents = [
            "python-requests/2.28.1",
            "curl/7.68.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MaliciousScanner/1.0",
            "CustomExfilTool/9.9"
        ]

        self.benign_actions = [
            "GetObject", "PutObject", "ListBuckets", "DescribeInstances",
            "StartInstances", "StopInstances", "GetSecretValue", "AssumeRole", "ListUsers"
        ]
        self.malicious_actions = [
            "AttachUserPolicy", "CreateAccessKey", "PutBucketPolicy",
            "DeleteCloudTrail", "UpdateAssumeRolePolicy", "CreateUser",
            "PutUserPolicy", "GetSecretValue"
        ]

    def generate_event(self, is_malicious: bool = False, base_time: datetime = None) -> Dict[str, Any]:
        """Generates a single synthetic CloudTrail event record."""
        if base_time is None:
            base_time = datetime.utcnow() - timedelta(days=7)
        
        # Random time offset within 7 days
        event_time = base_time + timedelta(
            seconds=random.randint(0, 7 * 24 * 3600)
        )

        if is_malicious:
            user = random.choice(self.malicious_users)
            ip = random.choice(self.malicious_ips)
            user_agent = random.choice(self.malicious_user_agents)
            action = random.choice(self.malicious_actions)
        else:
            user = random.choice(self.benign_users)
            ip = random.choice(self.benign_ips)
            user_agent = random.choice(self.benign_user_agents)
            action = random.choice(self.benign_actions)

        # Construct request parameters based on action
        request_params = {}
        if "Bucket" in action:
            request_params["bucketName"] = f"company-prod-data-{random.randint(1, 10)}"
            if action == "PutBucketPolicy":
                request_params["policy"] = '{"Statement":[{"Effect":"Allow","Principal":"*","Action":"s3:*","Resource":"*."}]}'
        elif "Policy" in action or "User" in action:
            request_params["userName"] = user["userName"]
            request_params["policyArn"] = "arn:aws:iam::aws:policy/AdministratorAccess"
        elif "Secret" in action:
            request_params["secretId"] = f"prod/db/credentials-{random.randint(1, 5)}"
        elif "Instances" in action:
            request_params["instanceId"] = f"i-{random.randint(10000000, 99999999)}"

        event = {
            "eventVersion": "1.08",
            "eventTime": event_time.isoformat() + "Z",
            "eventName": action,
            "eventSource": f"{action.lower().replace('get','').replace('put','').replace('create','')} .amazonaws.com",
            "userIdentity": user,
            "sourceIPAddress": ip,
            "userAgent": user_agent,
            "requestParameters": request_params,
            "is_malicious": 1 if is_malicious else 0
        }
        return event

    def generate_dataset(self, num_events: int = 2000, malicious_ratio: float = 0.15) -> List[Dict[str, Any]]:
        """Generates a dataset of CloudTrail events containing both benign and malicious logs."""
        num_malicious = int(num_events * malicious_ratio)
        num_benign = num_events - num_malicious

        events = []
        for _ in range(num_benign):
            events.append(self.generate_event(is_malicious=False))
        for _ in range(num_malicious):
            events.append(self.generate_event(is_malicious=True))

        # Shuffle events chronologically / randomly
        random.shuffle(events)
        return events

    def save_to_json(self, events: List[Dict[str, Any]], filepath: str = "cloudtrail_logs.json") -> None:
        """Saves generated events to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(events, f, indent=2)
        print(f"[+] Saved {len(events)} synthetic CloudTrail events to {filepath}")


if __name__ == "__main__":
    generator = CloudTrailDatasetGenerator(seed=42)
    logs = generator.generate_dataset(num_events=2000, malicious_ratio=0.15)
    generator.save_to_json(logs, "cloudtrail_logs.json")
