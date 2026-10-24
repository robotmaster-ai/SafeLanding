try:
    from viva import DroneEnv
    from viva import HMI
except ImportError:
    print("Warning: 'viva' module not found. Simulation will not run.")
    DroneEnv = None
    HMI = None

import numpy as np, pandas as pd, scipy, os, time, json, cv2
import scipy.signal # Explicit import for signal

import torch
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation

## SEGFORMER CONFIGURATION
SegFormer = 0 # 0 for B0, 5 for B5

## FILE CONFIGURATION
B0model_dir = "C:/Users/embat/Workspace/SegFormer/best_model.pth"
B5model_dir = "C:/Users/embat/Workspace/SegFormer/best_model_B5.pth"

c_dir = "C:/Users/embat/Datasets/SemanticDroneDataset/class_dict.csv"

video_dir = r"C:\Users\embat\Downloads\DJI_20250112133051_0013_D_alt50m\DJI_20250112133051_0013_D.MP4"
# video_dir = r"C:\Users\embat\Downloads\FNSM25_20250505181928_0031_D_alt30m\FNSM25_20250505181928_0031_D.MP4"
res_dir = r"C:\Users\embat\Downloads\vid"
# f_rhoX = 6.32692/4000.0
# f_rhoY = 4.7452/3000.0
f_rhoX = 4000/6.32692
f_rhoY = 3000/4.7452

# os.makedirs(dataset_dir, exist_ok=True)


## LOADING CONFIGURATION
if not os.path.exists(c_dir):
    # print("Adios")
    print(f"Warning: Class dict not found at {c_dir}.")
else:
    # print("Hola")
    color_map = pd.read_csv(c_dir, index_col=False, skipinitialspace=True)
    color_map.columns = ["label", "R", "G", "B", "risk"]

id2label = {id: label for id, label in enumerate(color_map.label)}
id2color = {id: [R,G,B] for id, (R,G,B) in enumerate(zip(color_map.R, color_map.G, color_map.B))}
id2risk = {id: risk for id, risk in enumerate(color_map.risk)}


def ssmask_RGB2code(rgb_labels, id2color):
    rgb_labels = np.array(rgb_labels)
    code_labels = np.zeros((rgb_labels.shape[0], rgb_labels.shape[1]), dtype=np.uint8)
    for id, color in id2color.items():
        code_labels[(rgb_labels == color).all(axis=-1)] = id

    return code_labels

def ssmask_code2RGB(code_img, id2color):
    rgb_img = np.zeros((code_img.shape[0], code_img.shape[1], 3), dtype=np.uint8)
    for id in id2color:
        rgb_img[code_img==id, :] = id2color[id]

    return rgb_img
    
def ssmask_RGB2risk(rgb_image, id2color, id2risk):
    rgb_image = np.array(rgb_image)
    code_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
    
    for id, color in id2color.items():
        code_image[(rgb_image == color).all(axis=-1)] = id2risk[id]
    
    return code_image

def ssmask_code2risk(code_image, id2risk):
    risk_image = np.zeros_like(code_image)
    
    for id, risk in id2risk.items():
        risk_image[code_image == id] = risk
    
    return risk_image


## TRANSFORMER DEFINITION
transform = transforms.Compose([transforms.ToTensor()])

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0" if SegFormer == 0 else "nvidia/mit-b5",
                                                         num_labels=24,
                                                         id2label=id2label, 
                                                         #label2id=label2id
                                                         )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_dir = B0model_dir if SegFormer == 0 else B5model_dir

if os.path.isfile(model_dir):
    print("Loading trained weights from '{}'".format(model_dir))
    checkpoint = torch.load(model_dir, map_location=device, weights_only=False)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print("{} model loaded from epoch {}".format("B5" if SegFormer == 5 else "B0", checkpoint['epoch']))
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded (no epoch info)")
    model.eval()
    print("Model ready for inference.")
else:
    print("No checkpoint found at '{}'. Using pretrained weights.".format(model_dir))


def main():
    if DroneEnv is None:
        print("Skipping simulation loop because 'viva' is missing.")
        return

    # Create an instance of the simulator environment
    env = DroneEnv(
        render_mode="human", 
        video=video_dir
    )

    # Create an instance of the HMI controller
    hmi = HMI()

    # Initialize the environment and drone pose
    obs, info = env.reset(psi_deg_init=0.0, z=15)
    terminated = False
    
    fr = []
    t0_f = time.time()
    f_i = 0 # Frame index
    
    #meter_inPixels = (round(f_rhoX / (video_data[0]["altitude"] / 1000)), round(f_rhoY / (video_data[0]["altitude"] / 1000)))
    meter_inPixels = (round(f_rhoX / (info["drone_state"].pos.z)), round(f_rhoY / (info["drone_state"].pos.z)))
    print(f_rhoX, f_rhoY, info["drone_state"].pos.z, meter_inPixels)

    # Main loop to run the simulation
    while not terminated:
        action, reset, terminated_command = hmi()
        obs, terminated, info = env.step(action)
        # print(info["drone_state"].pos.z)
        terminated = terminated or terminated_command
        if reset:
            obs, info = env.reset()


        #########################################################
        
        # Get frame from observation
        frame = obs.copy()
        
        with torch.inference_mode():

            t0_f = time.time()
            
            if frame.shape[0] > 1080:
                frame = cv2.resize(frame, (round(1080*(frame.shape[1]/frame.shape[0])), 1080), interpolation=cv2.INTER_AREA)
            elif frame.shape[1] > 1920:
                frame = cv2.resize(frame, (1920, round(1920*(frame.shape[0]/frame.shape[1]))), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensorframe = transform(frame).unsqueeze(0).to(device)

            # ==================================
            output = model(pixel_values=tensorframe)

            upsampled_logits = torch.nn.functional.interpolate(output.logits, size=frame.shape[:2], mode="bilinear", align_corners=False)
            sslogits = upsampled_logits.argmax(dim=1) + 1
            #sslogits = output.logits.argmax(dim=1) + 1
            sslogits[sslogits==255] = 0
            # ==================================

            
            vidframe = np.zeros((frame.shape[0]*2, frame.shape[1]*2, frame.shape[2]), dtype=np.uint8)
            

            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            vidframe[:frame.shape[0], :frame.shape[1], :] = frame
            
            
            # segframe = 0.5 * np.array(frame) + 0.5 * ssmask_code2RGB(sslogits.squeeze().cpu(), id2color)
            # cv2.imwrite(os.path.join(dataset_dir, "_" + image_name[f_i].split('.')[0] + "_1SemSeg.jpg"), segframe.astype(np.uint8))
            #
            segframe = ssmask_code2RGB(sslogits.squeeze().cpu(), id2color)
            vidframe[:frame.shape[0], frame.shape[1]:, :] = segframe.astype(np.uint8)
            

            riskframe = ssmask_code2risk(sslogits.squeeze().cpu(), id2risk)
            # segframe = 0.5 * np.array(frame) + 0.5 * cv2.applyColorMap((riskframe**3.44).astype(np.uint8), cv2.COLORMAP_HOT)
            # cv2.imwrite(os.path.join(dataset_dir, "_" + image_name[f_i].split('.')[0] + "_2Risk.jpg"), segframe.astype(np.uint8))
            #
            # segframe = cv2.applyColorMap((riskframe**3.44).astype(np.uint8), cv2.COLORMAP_HOT)
            segframe = cv2.applyColorMap((riskframe**3.44).astype(np.uint8), cv2.COLORMAP_HOT)
            vidframe[frame.shape[0]:, :frame.shape[1], :] = segframe.astype(np.uint8)
            
            
            # maxpooledframe = np.max(np.lib.stride_tricks.sliding_window_view(riskframe, (meter_inPixels[f_i], meter_inPixels[f_i])), axis=(2,3))
            maxpooledframe = np.max(np.lib.stride_tricks.sliding_window_view(riskframe, meter_inPixels), axis=(2,3))
            padding = ((frame.shape[0] - maxpooledframe.shape[0])//2, (frame.shape[1] - maxpooledframe.shape[1])//2)
            segframe = frame * 0.5
            segframe[padding[0]:maxpooledframe.shape[0]+padding[0], padding[1]:maxpooledframe.shape[1]+padding[1]] = 0.5 * cv2.applyColorMap((maxpooledframe**3.44).astype(np.uint8), cv2.COLORMAP_HOT)
            # segframe = 0.5 * np.array(frame) + segframe
            # cv2.imwrite(os.path.join(dataset_dir, "_" + image_name[f_i].split('.')[0] + "_3MaxP.jpg"), segframe.astype(np.uint8))
            #
            segframe = np.zeros_like(frame)
            segframe[padding[0]:maxpooledframe.shape[0]+padding[0], padding[1]:maxpooledframe.shape[1]+padding[1], :] = cv2.applyColorMap((maxpooledframe**3.44).astype(np.uint8), cv2.COLORMAP_HOT)
            

            #gaussians = (scipy.signal.gaussian(meter_inPixels[0]*2, meter_inPixels[0]/3), scipy.signal.gaussian(meter_inPixels[1]*2, meter_inPixels[1]/3))
            # gaussian = scipy.signal.gaussian(meter_inPixels[f_i]*2, meter_inPixels[f_i]/3)
            gaussian = scipy.signal.windows.gaussian(meter_inPixels[0]*2, meter_inPixels[0]/3)
            #gaussiankernel = np.outer(gaussians[0], gaussians[1]) / (2 * np.pi * ((meter_inPixels[0]/3)**2))
            # gaussiankernel = np.outer(gaussian, gaussian) / (2 * np.pi * ((meter_inPixels[f_i]/3)**2))
            gaussiankernel = np.outer(gaussian, gaussian) / (2 * np.pi * ((meter_inPixels[0]/3)**2))
            smoothedframe = scipy.signal.correlate(maxpooledframe**2, gaussiankernel, mode="valid")
            padding = ((frame.shape[0] - smoothedframe.shape[0])//2, (frame.shape[1] - smoothedframe.shape[1])//2)
            # segframe = 0.5 * np.array(frame)
            # segframe[padding[0]:smoothedframe.shape[0]+padding[0], padding[1]:smoothedframe.shape[1]+padding[1]] += 0.5 * cv2.applyColorMap((smoothedframe**1.72).astype(np.uint8), cv2.COLORMAP_HOT)
            # cv2.imwrite(os.path.join(dataset_dir, "_" + image_name[f_i].split('.')[0] + "_4RiskDis.jpg"), segframe.astype(np.uint8))
            #
            segframe = np.zeros_like(frame)
            segframe[padding[0]:smoothedframe.shape[0]+padding[0], padding[1]:smoothedframe.shape[1]+padding[1], :] = cv2.applyColorMap((smoothedframe**1.72).astype(np.uint8), cv2.COLORMAP_HOT)
            vidframe[frame.shape[0]:, frame.shape[1]:, :] = segframe
            

            vidframe = cv2.resize(vidframe, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA)
            # cv2.imwrite(os.path.join(dataset_dir, "_" + image_name[f_i].split('.')[0] + "_All.jpg"), vidframe)
            cv2.imwrite(os.path.join(res_dir, "img_" + str(t0_f) + "_All.jpg"), vidframe)

        fr.append(1/(time.time()-t0_f))
        
        # cv2.putText(vidframe, "FrameRate: {} FPS".format(fr[-1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (4, 8, 255))
        # cv2.imshow("SemanticSegmentation", vidframe)
        # cv2.waitKey(1)
        
        #print(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - t0)))
        print("Average Frame Rate: {} fps".format(np.average(fr)))
            
        # Add color legend boxes
        box_size = 20  # Medium-sized boxes
        padding = 5
        text_offset = 35
        start_x = frame.shape[1] + 10
        start_y = frame.shape[0] + 10
        
        for idx, (label, color) in enumerate(zip(color_map.label, zip(color_map.B, color_map.G, color_map.R))):
            y_pos = start_y + idx * (box_size + padding)
            # Draw colored box
            cv2.rectangle(vidframe, 
                            (start_x, y_pos), 
                            (start_x + box_size, y_pos + box_size), 
                            color, 
                            -1)  # Filled rectangle
            # Draw box border
            cv2.rectangle(vidframe, 
                            (start_x, y_pos), 
                            (start_x + box_size, y_pos + box_size), 
                            (255, 255, 255), 
                            1)  # White border
            # Add label text
            cv2.putText(vidframe, 
                        label, 
                        (start_x + text_offset, y_pos + box_size - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, 
                        (255, 255, 255), 
                        1, 
                        cv2.LINE_AA)
            
        # Display result (optional, if running locally with display)
        cv2.imshow("SegFormer", cv2.resize(vidframe, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fr.append(1/(time.time()-t0_f))
        t0_f = time.time()
        f_i += 1
        
    hmi.quit()

if __name__ == "__main__":
    main()