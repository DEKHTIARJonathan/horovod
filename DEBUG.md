# Debug Instructions

```bash
nvidia-docker run -it --rm \
    -e NVIDIA_VISIBLE_DEVICES=0,2 \
    --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd)/:/horovod/ \
    -v /media/jonathan/_dataSSD/datasets/imagenet/tfrecords/:/data/ \
    nvcr.io/nvidia/tensorflow:19.06-py3
```

## Build

`HOROVOD_WITH_TENSORFLOW=1 python setup.py install`

## Test
```bash
mpirun \
    -np 2 \
    -H localhost:2 \
    -bind-to none \
    -map-by slot \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -mca pml ob1 -mca btl ^openib \
    --allow-run-as-root \
    python examples/tensorflow_synthetic_benchmark.py
```