"""Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
from sklearn.datasets import fetch_kddcup99
import argparse
from tensorflow.io import gfile
import os
"""Script to create data to demonstrate TCAV on the KDD99 dataset.

Makes two example concepts. Creates folder with all targets in the dataset.
Creates 10 random folders. All generated data
will live under source_dir. TCAV will then use this directory to compute TCAV
scores.
Usage:

python make_kdd99_concepts.py --source_dir=YOUR_PATH

"""


def make_concepts_targets_and_randoms(source_dir):
  # Make concept folders with the csv files
  # We need this, since sklearn does not provide us with column names or types
  # categorical_variables = ["protocol_type", "service","flag","labels"]
  columns = [
      "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
      "land", "wrong_fragment", "urgent", "hot", "m_failed_logins", "logged_in",
      "num_compromised", "root_shell", "su_attempted", "num_root",
      "num_file_creations", "num_shells", "num_access_files",
      "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
      "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
      "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
      "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
      "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
      "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
      "dst_host_srv_serror_rate", "dst_host_rerror_rate",
      "dst_host_srv_rerror_rate", "label"
  ]

  data, labels = fetch_kddcup99(return_X_y=True)

  # Create dataframe from the KDD dataset
  dataset_columns = {columns[i]: data[:, i] for i in range(len(data[0]))}
  label_columns = {"labels": labels}
  dataset_columns.update(label_columns)
  dataframe = pd.DataFrame(dataset_columns)

  def make_concept_folder(dataframe, concept):
    # Create the folder and save the dataframe as a csv file there
    path = os.path.join(source_dir, concept)
    if not gfile.exists(path):
      gfile.makedirs(path)

    concept_file_name = os.path.join(path, concept + ".csv")
    dataframe.to_csv(concept_file_name, index=False)

  concept_less_df = dataframe[dataframe["dst_host_same_src_port_rate"] < 1]
  concept_more_df = dataframe[dataframe["dst_host_same_src_port_rate"] >= 1]

  make_concept_folder(concept_less_df,
                      "dst_host_same_src_port_rate_less_than_one")
  make_concept_folder(concept_more_df,
                      "dst_host_same_src_port_rate_more_than_one")

  # Making random_examples
  random_size = 10
  random_partitions = 11

  for i in range(random_partitions):
    random_partition_name = "random500_" + str(i)
    randoms = dataframe.sample(random_size)
    make_concept_folder(randoms, random_partition_name)

  # Make target folder
  targets = dataframe.labels.unique()
  print("Available concepts for KDD99 dataset are: ")
  print([
      "dst_host_same_src_port_rate_less_than_one",
      "dst_host_same_src_port_rate_more_than_one"
  ])
  print("\n")
  print("Available targets for KDD99 dataset are: ")
  print(targets)
  print("\n")
  print("Created 10 random folders \n")
  for target in targets:
    target_df = dataframe[dataframe["labels"] == target]
    make_concept_folder(target_df, target.decode("utf-8"))

  # make labels
  with open(os.path.join(source_dir, "labels.txt"), "w") as the_file:
    for target in targets:
      the_file.write(target.decode("utf-8") + "\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Create data folders for the KDD99 model.")
  parser.add_argument(
      "--source_dir",
      type=str,
      help="Name for the directory where we will save the data.")

  args = parser.parse_args()
  # create folder if it doesn't exist
  if not gfile.exists(args.source_dir):
    gfile.makedirs(os.path.join(args.source_dir))
    print("Created source directory at " + args.source_dir)
  # Make data
  make_concepts_targets_and_randoms(args.source_dir)
  print("Successfully created data at " + args.source_dir)
