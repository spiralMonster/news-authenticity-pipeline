import yaml


def replace_paths(obj,old_root,new_root):
    if isinstance(obj,dict):
        return {
            k:replace_paths(v,old_root,new_root)
            for k,v in obj.items()
        }

    if isinstance(obj,list):
        return [
            replace_paths(v,old_root,new_root)
            for v in obj
        ]

    if isinstance(obj,str):
        return obj.replace(old_root,new_root)

    return obj



def PatchPipelineYaml(
        yaml_path:str,
        pvc_name:str,
        local_mount_path:str,
        kubernetes_mount_path:str


):
    with open(yaml_path, "r") as file:
        pipeline_spec = yaml.safe_load(file)

    pipeline_spec["schemaVersion"] = "2.1.0"

    volume = {
        "name": "kubeflow-pv",
        "persistentVolumeClaim": {
            "claimName": pvc_name
        },
    }

    volume_mount = {
        "name": "kubeflow-pv",
        "mountPath":kubernetes_mount_path
    }

    executors = pipeline_spec.get("deploymentSpec", {}).get("executors", {})

    for executor in executors.values():
        container = executor.get("container")
        if container is None:
            continue

        volume_mounts = container.setdefault("volumeMounts", [])
        if not any(vm.get("name") == "kubeflow-pv" for vm in volume_mounts):
            volume_mounts.append(volume_mount)

        volumes = executor.setdefault("volumes", [])
        if not any(v.get("name") == "kubeflow-pv" for v in volumes):
            volumes.append(volume)



    pipeline_spec=replace_paths(
        pipeline_spec,
        old_root=local_mount_path,
        new_root=kubernetes_mount_path
    )

    with open(yaml_path, "w") as file:
        yaml.safe_dump(pipeline_spec, file, sort_keys=False)