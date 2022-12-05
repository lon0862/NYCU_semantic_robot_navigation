import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import argparse
import copy
from decimal import Decimal
import math
import shutil
import os

# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "../habitat-lab/replica_v1/apartment_0/habitat/mesh_semantic.ply"
path = "../habitat-lab/replica_v1/apartment_0/habitat/info_semantic.json"

# default
txt_filename = "path_output/world_coordinate/refrigerator.txt"
img_path = "path_output/map_img/refrigerator.png"
save_dir = "path_output/RGB_img/refrigerator/"
save_num = 0

#global test_pic
#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    
    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def parse_args():
    parser = argparse.ArgumentParser(description='define target object')
    parser.add_argument('--target', type=str, default='refrigerator')
    args = parser.parse_args()
    return args

def initTarget(target):
    label_dict = {}
    label_dict["refrigerator"] = 67
    label_dict["rack"] = 66
    label_dict["cushion"] = 29
    label_dict["lamp"] = 47
    label_dict["cooktop"] = 32

    if target in label_dict:
        target_label = label_dict[args.target]
    else:
        print("object is not right, default using refrigerator")
        target_label = target_dict["refrigerator"]
    
    return target_label

def highlight(observations, target_label):
    obs_label = id_to_label[observations["semantic_sensor"]]
    img_color = observations["color_sensor"] # (r,g,b)
    mark_color = copy.deepcopy(img_color)
    for i in range(obs_label.shape[0]):
        for j in range(obs_label.shape[1]):
            if(obs_label[i][j]==target_label):
                mark_color[i][j] = np.array([255,0,0,150])
            else:
                mark_color[i][j] = np.array([0,0,0,0])

    mark_img = cv2.add(img_color, mark_color)
    return mark_img

def rotate_one_time(target_label, sim, agent, action_names, degree, slide, action):
     # do relative action
    loop = int(degree)//slide
    for step in range(loop):
        sim.config.agents[0].action_space[action] = habitat_sim.agent.ActionSpec(
                action, habitat_sim.agent.ActuationSpec(amount=slide))
        navigateAndSee(action, target_label, sim, agent, action_names)

    remain_degree = degree - loop*slide
    sim.config.agents[0].action_space[action] = habitat_sim.agent.ActionSpec(
                action, habitat_sim.agent.ActuationSpec(amount=remain_degree))
    navigateAndSee(action, target_label, sim, agent, action_names)

def move(data, target_label, sim, agent, action_names):
    origin_degree = 0
    slide = 15
    # reach last point before goal
    for i in range(len(data)-1):
        x = round(data[i+1][0]-data[i][0], 3)
        z = round(data[i+1][1]-data[i][1], 3)
        dist = math.sqrt(x**2 + z**2)
        pre_degree = origin_degree
        if(x>0): 
            radian = math.atan(z/(-1*x))
            degree = 90-math.degrees(radian)
            origin_action = "turn_right"
            origin_degree = degree
            degree+=pre_degree

            if(degree<0):
                degree*=-1
                action = "turn_left"
                rotate_one_time(target_label, sim, agent, action_names, degree, slide, action)
            else:
                action = "turn_right"
                rotate_one_time(target_label, sim, agent, action_names, degree, slide, action)
        else:
            if(x==0):
                if(z<0):
                    degree=0
                else:
                    degree=180
            else:
                radian = math.atan(z/x)
                degree = 90-math.degrees(radian)
            origin_action = "turn_left"
            origin_degree = degree
            degree-=pre_degree
            if(degree<0):
                degree*=-1
                action = "turn_right"
                rotate_one_time(target_label, sim, agent, action_names, degree, slide, action)
            else:
                action = "turn_left"
                rotate_one_time(target_label, sim, agent, action_names, degree, slide, action)

        sim.config.agents[0].action_space["move_forward"] = habitat_sim.agent.ActionSpec(
                    "move_forward", habitat_sim.agent.ActuationSpec(amount=dist))
        
        # At last, only forward to goal, no move_forward again
        if(i<len(data)-2):
            navigateAndSee("move_forward", target_label, sim, agent, action_names)
            # right degree seem as negative
            if(origin_action=="turn_right"):
                origin_degree*=-1

def navigateAndSee(action, target_label, sim, agent, action_names):
    global save_dir
    global save_num
    if action in action_names:
        observations = sim.step(action)
        mark_img = highlight(observations, target_label)
        save_path = save_dir + str(save_num) + ".png"
        cv2.imwrite(save_path, transform_rgb_bgr(mark_img))
        save_num+=1

        if(action == "move_forward"):
            cv2.imshow("map and RGB", transform_rgb_bgr(mark_img))
            cv2.waitKey(1)
            agent_state = agent.get_state()
            sensor_state = agent_state.sensor_states['color_sensor']
            print("camera pose: x y z rw rx ry rz")
            print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        
def img_to_video(target, map_path, img_dir):
    size = (1024,512)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = "path_output/video/"+target+".mp4"
    videowrite = cv2.VideoWriter(video_path,fourcc,3,size)
    img_array=[]
    file_num = len([name for name in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, name))])
    for i in range(file_num):
        img_path = img_dir + str(i) + ".png"
        map_img = cv2.imread(map_path)
        rgb_img = cv2.imread(img_path)
        imgs = np.hstack([map_img, rgb_img])
        videowrite.write(imgs)

    print("video save in", video_path)

if __name__ == "__main__":
    args = parse_args()
    target = args.target
    target_label = initTarget(target)
    txt_filename = "path_output/world_coordinate/"+target+".txt"
    img_path = "path_output/map_img/"+target+".png"
    save_dir = "path_output/RGB_img/"+target+"/"
    with open(txt_filename, 'r') as f:
        data = f.read()
        data = eval(data)
    
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    # Set agent state
    agent_state = habitat_sim.AgentState()
    # init agent position in world space, y=0.0 is floor1, origin is [0,0,0]
    agent_state.position = np.array([data[0][0], 0.0, data[0][1]])
    agent.set_state(agent_state)
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

    if(os.path.isdir(save_dir)):
        shutil.rmtree(save_dir)

    os.mkdir(save_dir)
    move(data, target_label, sim, agent, action_names)
    img_to_video(target, img_path, save_dir)
    