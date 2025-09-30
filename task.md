you are operating in the multi agent docker on our host system

> docker ps
CONTAINER ID   IMAGE                                                         COMMAND                  CREATED             STATUS                         PORTS                                                                                                                                                       NAMES
76d772e1c7eb   ar-ai-knowledge-graph-webxr                                   "./dev-entrypoint.sh"    10 minutes ago      Up 10 minutes                  4000/tcp, 5173/tcp, 0.0.0.0:3001->3001/tcp, [::]:3001->3001/tcp, 24678/tcp                                                                                  visionflow_container
13c460906ba5   cloudflare/cloudflared:latest                                 "cloudflared --no-au…"   About an hour ago   Up 10 minutes                                                                                                                                                                              cloudflared-tunnel
ca431834ad1d   multi-agent-docker:latest                                     "/entrypoint.sh /bin…"   About an hour ago   Up 59 minutes                  0.0.0.0:3000->3000/tcp, [::]:3000->3000/tcp, 0.0.0.0:3002->3002/tcp, [::]:3002->3002/tcp, 0.0.0.0:9500-9503->9500-9503/tcp, [::]:9500-9503->9500-9503/tcp   multi-agent-container
56e863d5884b   gui-tools-docker:latest                                       "/home/blender/start…"   About an hour ago   Up About an hour (unhealthy)   0.0.0.0:5901->5901/tcp, [::]:5901->5901/tcp, 0.0.0.0:9876-9879->9876-9879/tcp, [::]:9876-9879->9876-9879/tcp, 9222/tcp                                      gui-tools-container
d398fef838d8   swr.cn-north-4.myhuaweicloud.com/infiniflow/ragflow:nightly   "./entrypoint.sh"        32 hours ago        Up 32 hours                    0.0.0.0:80->80/tcp, [::]:80->80/tcp, 0.0.0.0:443->443/tcp, [::]:443->443/tcp, 0.0.0.0:9380->9380/tcp, [::]:9380->9380/tcp                                   ragflow-server
b5e2eb63ee40   elasticsearch:8.11.3                                          "/bin/tini -- /usr/l…"   32 hours ago        Up 32 hours (healthy)          9300/tcp, 0.0.0.0:1200->9200/tcp, [::]:1200->9200/tcp                                                                                                       ragflow-es-01
effb8f419ed8   valkey/valkey:8                                               "docker-entrypoint.s…"   32 hours ago        Up 32 hours (healthy)          0.0.0.0:6379->6379/tcp, [::]:6379->6379/tcp                                                                                                                 ragflow-redis
46f024bea631   quay.io/minio/minio:RELEASE.2025-06-13T11-33-47Z              "/usr/bin/docker-ent…"   32 hours ago        Up 32 hours (healthy)          0.0.0.0:9000-9001->9000-9001/tcp, [::]:9000-9001->9000-9001/tcp                                                                                             ragflow-minio
79ad81d0246e   mysql:8.0.39                                                  "docker-entrypoint.s…"   32 hours ago        Up 32 hours (healthy)          33060/tcp, 0.0.0.0:5455->3306/tcp, [::]:5455->3306/tcp                                                                                                      ragflow-mysql
ec342ef367ac   registry.gitlab.com/aadnk/whisper-webui:latest                "python3 app.py --in…"   9 days ago          Up 9 days                      0.0.0.0:7860->7860/tcp, [::]:7860->7860/tcp                                                                                                                 whisper-webui
5698461a6765   ghcr.io/remsky/kokoro-fastapi-gpu:latest                      "/opt/nvidia/nvidia_…"   12 days ago         Up 12 days                     0.0.0.0:8880->8880/tcp, [::]:8880->8880/tcp                                                                                                                 kokoro-tts-container
>  docker stop 13c460906ba5
13c460906ba5
> docker network inspect docker_ragflow
[
    {
        "Name": "docker_ragflow",
        "Id": "b0c38a1301451c0329969ef53fdedde5221b1b05b063ad94d66017a45d3ddaa3",
        "Created": "2025-04-05T14:36:31.500965678Z",
        "Scope": "local",
        "Driver": "bridge",
        "EnableIPv4": true,
        "EnableIPv6": false,
        "IPAM": {
            "Driver": "default",
            "Options": null,
            "Config": [
                {
                    "Subnet": "172.18.0.0/16",
                    "Gateway": "172.18.0.1"
                }
            ]
        },
        "Internal": false,
        "Attachable": false,
        "Ingress": false,
        "ConfigFrom": {
            "Network": ""
        },
        "ConfigOnly": false,
        "Containers": {
            "46f024bea631306afe91f7a7cf75113820019fca36e3842fd51263b33e4fe881": {
                "Name": "ragflow-minio",
                "EndpointID": "f012e5c1cdeec905da201ef35010cb8867cec6c7de3450c2667f4180cfa02bcd",
                "MacAddress": "82:b0:a6:c8:d1:8b",
                "IPv4Address": "172.18.0.2/16",
                "IPv6Address": ""
            },
            "5698461a67654b6428f8b6ae9086d05d3cba433e6178f8e80c854d81ef3b1a67": {
                "Name": "kokoro-tts-container",
                "EndpointID": "adac268a8919fa9bad016bcecfe19eecaf662a42a78cbf847791218b872c83e8",
                "MacAddress": "0a:94:07:1d:88:29",
                "IPv4Address": "172.18.0.5/16",
                "IPv6Address": ""
            },
            "56e863d5884bf991e1f6e3af73095e4e6ec7e1a7303e6f12f289852cfef65c82": {
                "Name": "gui-tools-container",
                "EndpointID": "bae022f4f810a776577a41720690eeda0ae46e60de52517340ca5cd0d60eb9b4",
                "MacAddress": "8e:cd:30:c4:b6:db",
                "IPv4Address": "172.18.0.4/16",
                "IPv6Address": ""
            },
            "76d772e1c7eb0c8883e3c9abd837fd31150086bd5e1ccf1a94ed29404a46d546": {
                "Name": "visionflow_container",
                "EndpointID": "2d21f7cb8fd0950c802cfaf4dd8c0d4331e833a66f206c57bdda8e6396beb3f7",
                "MacAddress": "ce:a7:d6:e4:90:3e",
                "IPv4Address": "172.18.0.11/16",
                "IPv6Address": ""
            },
            "79ad81d0246e64439471148534abab577005f052248550108963ce3ef8a4e14b": {
                "Name": "ragflow-mysql",
                "EndpointID": "46fc44b401bef69e840b63ae89f2f6a141d5d6223a5a96b421f5306d4f0e1608",
                "MacAddress": "ca:f0:79:12:df:f7",
                "IPv4Address": "172.18.0.8/16",
                "IPv6Address": ""
            },
            "b5e2eb63ee40e3ea0bf315d0f04d95dac20ee31d4199bcf4cf698519115f4e1f": {
                "Name": "ragflow-es-01",
                "EndpointID": "fa2353342e174059daf37fbab03531de6f4665116abac0ecc2152a5b47932881",
                "MacAddress": "ce:ff:1f:86:9f:bc",
                "IPv4Address": "172.18.0.7/16",
                "IPv6Address": ""
            },
            "ca431834ad1d561df88a042e8157bb03e955370b8ca52c506f2e7cc0675e461d": {
                "Name": "multi-agent-container",
                "EndpointID": "9ef213c3ac58d4e7d63a862f971e0dc2cfcfcbf4d5f1ad2c03d86246980a1c29",
                "MacAddress": "fe:c8:30:2a:48:6a",
                "IPv4Address": "172.18.0.9/16",
                "IPv6Address": ""
            },
            "d398fef838d88269be503d379cf8220813944e2c665b211c7defbb70cda106c9": {
                "Name": "ragflow-server",
                "EndpointID": "f2da61ab0b2edd39ed52356dd4a54d1272f5d213b2aed2e2bf72fcda1539e5f6",
                "MacAddress": "36:55:19:f7:4a:91",
                "IPv4Address": "172.18.0.10/16",
                "IPv6Address": ""
            },
            "ec342ef367ace011e1390e53298ff2b37d6397e20f560af44b066f58142ff4ae": {
                "Name": "whisper-webui",
                "EndpointID": "9ac164a4f5b19f5c5d1a06a2045b45c226b8047867acac97c8fffda7b22f90f8",
                "MacAddress": "ea:3d:8a:17:7c:7c",
                "IPv4Address": "172.18.0.3/16",
                "IPv6Address": ""
            },
            "effb8f419ed82cf4bebc0f2705e3faea36ec098d5dadfcd6d81f492e72ead434": {
                "Name": "ragflow-redis",
                "EndpointID": "729b4f0b3fbe7e56ec9c76920d3f21f2119b16fd6e76a54c1b90053f058f0a22",
                "MacAddress": "8a:4d:ee:a2:cd:7b",
                "IPv4Address": "172.18.0.6/16",
                "IPv6Address": ""
            }
        },
        "Options": {},
        "Labels": {
            "com.docker.compose.config-hash": "20de4b714cebc3288cab9ac5bf17cbed67f64545e9b273c2e547d4a6538609b9",
            "com.docker.compose.network": "ragflow",
            "com.docker.compose.project": "docker",
            "com.docker.compose.version": "2.34.0"
        }
    }
]

you cannot see the operations of the docker project described in ext/src but you can confirm runtime vs the ext/multi-agent-docker project which is OUR container.

your task is to fully consolidate new semi structured updates containing in ext/docs into the structured corpus of the docs system, using best in class systems and diagrams and conventions. Maximise the available information. Where you find that some information in one file is at odds with another you can check the client code in ext/client the server code in ext/src and the multi-agent container mcp and agent stuff in ext/multi-agent-docker.  You should only update in ext/docs, removing legacy and incorrect material, integrating into the relevant locations and confirming against code. use uk english spelling.

work using a swarm until everything is done and old files are cleaned up, we have version control so don't worry about deleting within ext/docs

remove all reports about the last cycle of development, they are everywhere. we should capture the NOW using descriptive best practice, discarding the journey.