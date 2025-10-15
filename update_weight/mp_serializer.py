# Adapted from https://github.com/sgl-project/sglang/blob/3b3b3baf9f08a8f2e6180c9f9146b6137ad8032c/python/sglang/srt/utils.py#L1846-L1885
#
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import io
import pybase64
from multiprocessing.reduction import ForkingPickler

class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """
        Serialize a Python object using ForkingPickler.

        Args:
            obj: The object to serialize.
            output_str (bool): If True, return a base64-encoded string instead of raw bytes.

        Returns:
            bytes or str: The serialized object.
        """
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            # Convert bytes to base64-encoded string
            output = pybase64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        """
        Deserialize a previously serialized object.

        Args:
            data (bytes or str): The serialized data, optionally base64-encoded.

        Returns:
            The deserialized Python object.
        """
        if isinstance(data, str):
            # Decode base64 string to bytes
            data = pybase64.b64decode(data, validate=True)

        return ForkingPickler.loads(data)
