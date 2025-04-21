"""
openpi_server.py

Serves a policy using the websocket protocol, matching the openpi policy server interface.

To use it, first install:

```
pip install websockets
cd $OPENPI_ROOT/packages/openpi-client && pip install -e .
```

Then run the server:

```
python vla-scripts/openpi_server.py --policy $POLICY_CHECKPOINT_PATH
```


"""

import argparse
import asyncio
import logging
import traceback
import os

os.environ["PRISMATIC_DATA_ROOT"] = ""   # set to dummy value to prevent error

import numpy as np
import PIL.Image as Image
import torch
import websockets.asyncio.server
import websockets.frames
from openpi_client import msgpack_numpy

from prismatic.models.load import load_vla
from prismatic.vla.action_tokenizer import FASTTokenizer


def run_policy(vla, obs):
    def resize_image(image):
        return image.resize((224, 224), Image.Resampling.LANCZOS)

    action = vla.predict_action(
        [
            resize_image(Image.fromarray(obs["observation/exterior_image_1_left"]).convert("RGB")),
            resize_image(Image.fromarray(obs["observation/wrist_image_left"]).convert("RGB"))
        ],
        # resize_image(Image.fromarray(obs["observation/exterior_image_1_left"]).convert("RGB")),
        obs["prompt"].lower(),
        proprio=np.concatenate(
            [
                obs["observation/joint_position"],
                obs["observation/gripper_position"]
            ]
        ),
        max_tokens=128,
    )
    return action


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        # Load the VLA model
        vla = load_vla(policy_path, image_sequence_len=2)

        # Cast to half precision, move to GPU.
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
        vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
        vla.to(dtype=vla.llm_backbone.half_precision_dtype)
        vla.to(device)

        # For FAST tokenizer, initialize with one forward pass
        if isinstance(vla.action_tokenizer, FASTTokenizer):
            vla.action_tokenizer(np.zeros((1, 16, 8)))

        self._policy = vla
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        # import glob
        # import pickle
        # import simplejpeg

        # DATA_PATH = "/app/karl/datasets/droid_raw"
        # data_files = glob.glob(os.path.join(DATA_PATH, "*.pkl"))
        # data_files.sort()
        # print(f"Loaded {len(data_files)} files")

        # with open("/home/karl/code/openpi/actions_100_monopi.pkl", "rb") as f:
        #     gt_actions = pickle.load(f)

        # mses = []
        # for i, file_path in enumerate(data_files[:100]):
        #     with open(file_path, "rb") as f:
        #         data = pickle.load(f)
        #     exterior_image = simplejpeg.decode_jpeg(data["image_base"])
        #     wrist_image = simplejpeg.decode_jpeg(data["image_wrist"])
        #     obs = {
        #         "observation/exterior_image_1_left": np.array(exterior_image),
        #         "observation/wrist_image_left": np.array(wrist_image),
        #         "observation/joint_position": data["state"][:7],
        #         "observation/gripper_position": data["state"][-1:],
        #         "prompt": str(data["instruction"]),
        #     }
        #     gt_action = data["action"]
        #     gt_action = gt_actions[i]
        #     if gt_action.shape[0] < 10:
        #         continue
        #     action = run_policy(self._policy, obs)
        #     if np.linalg.norm(action[0] - action[-1]) < 1e-6:
        #         # Skip mis-decoded actions
        #         continue
        #     mse = np.mean((action[:10] - gt_action[:10]) ** 2)
        #     mses.append(mse)
        #     logging.info(f"MSE: {mse}")

        # logging.info(f"\nAverage MSE: {np.mean(mses)}")
        # exit(0)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                action = run_policy(self._policy, obs)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    args = parser.parse_args()
    server = WebsocketPolicyServer(args.policy)
    server.serve_forever()
