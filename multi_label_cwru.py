import pandas as pd
import os
import random
import shutil

class MultiLabelSplitter:
    def __init__(self, cwru_root_path):
        self.root_path = cwru_root_path

        self.paths = {
            "DE": os.path.join(
                cwru_root_path,
                "12k_Drive_End_Bearing_Fault_Data"
            ),

            "FE": os.path.join(
                cwru_root_path,
                "12k_Fan_End_Bearing_Fault_Data"
            )
        }

        self.normal_path = os.path.join(
            cwru_root_path,
            "Normal"
        )

        self.label_map = {
            "IR": [1, 0, 0],
            "OR": [0, 1, 0],
            "B":  [0, 0, 1]
        }

        self.valid_severities = ["007", "014", "021"]

    def create_folders(self, output_path):

        fault_types = ["B", "IR", "OR"]
        locations = ["DE", "FE"]
        splits = ["train", "test"]

        for fault in fault_types:
            for location in locations:
                for split in splits:

                    folder = os.path.join(
                        output_path,
                        fault,
                        location,
                        split
                    )

                    os.makedirs(folder, exist_ok=True)

        # Normal/test
        os.makedirs(
            os.path.join(output_path, "Normal", "test"),
            exist_ok=True
        )

    def _valid_or(self, severity, root, severity_path):
        if severity in ["007", "021"]:
            return "@6" in root

        if severity == "014":

            is_direct_014 = root == severity_path

            is_at6 = "@6" in root

            return is_direct_014 or is_at6

        return False

    def filtering(self):

        data = []

        for location, root_path in self.paths.items():

            for fault_type in os.listdir(root_path):

                if fault_type not in self.label_map:
                    continue

                fault_path = os.path.join(
                    root_path,
                    fault_type
                )

                if not os.path.isdir(fault_path):
                    continue

                for severity in os.listdir(fault_path):

                    if severity not in self.valid_severities:
                        continue

                    severity_path = os.path.join(
                        fault_path,
                        severity
                    )

                    if not os.path.isdir(severity_path):
                        continue

                    for root, _, files in os.walk(severity_path):

                        if fault_type == "OR":
                            if not self._valid_or(
                                severity,
                                root,
                                severity_path
                            ):
                                continue

                        for file in files:

                            if not file.endswith(".mat"):
                                continue

                            data.append({
                                "path": os.path.join(root, file),
                                "label": self.label_map[fault_type],
                                "type": fault_type,
                                "severity": severity,
                                "location": location
                            })

        df = pd.DataFrame(data)

        print(f"Total files: {len(df)}")

        random.seed(42)

        buckets = [
            ("DE", "IR"),
            ("DE", "OR"),
            ("DE", "B"),
            ("FE", "IR"),
            ("FE", "OR"),
            ("FE", "B")
        ]

        recipe = {
            bucket: random.choice(self.valid_severities)
            for bucket in buckets
        }

        print("\nSplit Recipe")
        print("-" * 30)

        for (loc, fault), sev in recipe.items():
            print(f"{loc}-{fault} -> {sev}")

        test_mask = pd.Series(False, index=df.index)

        for (loc, fault), sev in recipe.items():

            test_mask |= (
                (df["location"] == loc) &
                (df["type"] == fault) &
                (df["severity"] == sev)
            )

        train_df = df[~test_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        print("\nDataset Summary")
        print("-" * 30)
        print(f"Train: {len(train_df)}")
        print(f"Test : {len(test_df)}")

        return train_df, test_df

    def splitting(self, output_path, df, train=True):

        split = "train" if train else "test"

        for _, row in df.iterrows():

            dst = os.path.join(
                output_path,
                row["type"],
                row["location"],
                split
            )

            os.makedirs(dst, exist_ok=True)

            shutil.copy(
                row["path"],
                dst
            )

    def move_normal(self, output_path):

        normal_test_path = os.path.join(
            output_path,
            "Normal",
            "test"
        )

        for root, _, files in os.walk(self.normal_path):

            for file in files:

                if file.endswith(".mat"):

                    shutil.copy(
                        os.path.join(root, file),
                        normal_test_path
                    )