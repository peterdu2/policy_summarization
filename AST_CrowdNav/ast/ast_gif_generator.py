import os
import subprocess


if __name__ == '__main__':
    render_dir = '/home/peter/policy_summarization/AST_CrowdNav/ast/results/data/random_sample_data/human_position_set_10/AC'
    palette_path = '/home/peter/policy_summarization/AST_CrowdNav/ast/palette.png'
    output_dir = '/home/peter/policy_summarization/AST_CrowdNav/ast/results/data/random_sample_data/human_position_set_10/AC/gifs'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trajectories = [x[0] for x in os.walk(render_dir)]

    for traj_dir in trajectories:
        image_filenames = traj_dir + r'/%04d.png'
        gif_name = traj_dir[-3:]
        output_path = output_dir + '/' + str(gif_name).zfill(3) + '.gif'

        gif_gen_cmd = 'ffmpeg -y -framerate 7 -i ' + image_filenames + ' -i ' + palette_path + ' -lavfi paletteuse ' + output_path

        print('Running: ', gif_gen_cmd)
        subprocess.call(gif_gen_cmd, shell=True)

