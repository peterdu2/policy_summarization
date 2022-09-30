import os
import subprocess


if __name__ == '__main__':
    render_dir = '/home/peterdu2/policy_summarization/AST_CrowdNav/ast/results/data/ast_dsrnn_0/renders'
    palette_path = '/home/peterdu2/policy_summarization/AST_CrowdNav/ast/palette.png'
    output_dir = '/home/peterdu2/policy_summarization/AST_CrowdNav/ast/results/data/ast_dsrnn_0/renders/gifs'
    trajectories = [x[0] for x in os.walk(render_dir)]

    # subprocess.call("echo Hello World", shell=True)
    for traj_dir in trajectories:
        input_filenames_string = traj_dir + r'/%04d.png'
        gif_name = traj_dir[-3:]
        output_gif_name_string = output_dir + '/' + str(gif_name).zfill(3) + '.gif'
        #palletegen_cmd = 'ffmpeg -y -i ' + filename_string + ' -vf palettegen palette.png'

        gif_gen_cmd = 'ffmpeg -y -framerate 5 -i ' + input_filenames_string + ' -i ' + palette_path + ' -lavfi paletteuse ' + output_gif_name_string
        # print(' ')
        # print(gif_gen_cmd)
        subprocess.call(gif_gen_cmd, shell=True)
        #subprocess.call(palletegen_cmd, shell=True)
