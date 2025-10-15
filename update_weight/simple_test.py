def send_func():
    import torch
    import os
    from serialize_state_dict import state_dict_to_serialized_flattened_tensor

    torch.cuda.set_device(0)

    state_dict = {"float": torch.arange(0, 100, device="cuda").reshape(10, 10).float(),
                  "int": torch.arange(0, 32, device="cuda").reshape(4, 8).int(),
                  "bool": torch.ones(50, device="cuda").reshape(5, 10).bool()}

    serialized = state_dict_to_serialized_flattened_tensor(state_dict, output_str=True)
    with open("./ipc_handle.txt", "w") as f:
        f.write(serialized)
        print(f"[Send] serialized: {serialized}")
    while True:
        if os.path.exists("./ipc_transfered.txt"):
            os.remove("./ipc_transfered.txt")
            break
    print("[Send] Sender finished")
    return


def receive_func():
    import torch
    import os
    from serialize_state_dict import serialized_flattened_tensor_to_state_dict

    torch.cuda.set_device(0)

    while True:
        if os.path.exists("./ipc_handle.txt"):
            with open("./ipc_handle.txt", "r") as f:
                serialized = f.read()
            os.remove("./ipc_handle.txt")
            break

    state_dict = serialized_flattened_tensor_to_state_dict(serialized)
    assert torch.equal(state_dict["float"], torch.arange(0, 100, device="cuda").reshape(10, 10).float())
    assert torch.equal(state_dict["int"], torch.arange(0, 32, device="cuda").reshape(4, 8).int())
    assert torch.equal(state_dict["bool"], torch.ones(50, device="cuda").reshape(5, 10).bool())
    print(f"[Recv] float: {state_dict['float']}")
    print(f"[Recv] int: {state_dict['int']}")
    print(f"[Recv] bool: {state_dict['bool']}")
    with open("./ipc_transfered.txt", "w") as f:
        f.write("done")
    print(f"[Recv] Receiver finished")
    return

def main():
    import multiprocessing

    p1 = multiprocessing.Process(target=send_func)
    p2 = multiprocessing.Process(target=receive_func)

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("Test finished")

if __name__ == "__main__":
    main()
