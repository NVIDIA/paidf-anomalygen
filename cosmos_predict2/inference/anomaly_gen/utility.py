# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cosmos_predict2.utils.config_helper import get_config_module, override
from cosmos_predict2.scripts.anomaly_gen.ad_train import set_nested_attributes
import importlib
import yaml

COSMOS_ANOMALY_CONFIG = "cosmos_predict2/configs/base/ag_config.py"

def set_config(ad_checkpoint_dir, override_opts = ["--", "experiment=text2anomaly_7b"]):
    config_module = get_config_module(COSMOS_ANOMALY_CONFIG)
    config = importlib.import_module(config_module).make_config()
    config = override(config, override_opts)

    # Load config from pretrained checkpoint
    with open(f"{ad_checkpoint_dir}/ad_config.yaml") as fp:
        try:
            ad_config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"[ERROR] Cannot load {ad_checkpoint_dir}/ad_config.yaml file! Exception: {exc}")    

    # Merge config w/ ad_config
    set_nested_attributes(config, ad_config)
    return configc
