import requests
import os
import sys

def download_file(path, url):
    r = requests.get(url, stream=True)
    local_filename = url.split('/')[-1].split('?')[0]
    with open(os.path.join(path, local_filename), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)
    return local_filename


def download_models():
    directory = '_models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    dis_url = 'https://www.dropbox.com/s/tg738ajgc6o9u84/WGANDiscriminator.npz?dl=1'
    gen_url = 'https://www.dropbox.com/s/wzbig2iald8yk2s/DCGANGenerator.npz?dl=1'

    download_file(directory, dis_url)
    download_file(directory, gen_url)

def download_iq_models():
    directory = '_models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    dis_url = 'https://www.dropbox.com/s/p7wvkuygu91g87v/WGANDiscriminator_iq.npz?dl=1'
    gen_url = 'https://www.dropbox.com/s/fq0wcx8ldksn1n8/DCGANGenerator_iq.npz?dl=1'

    download_file(directory, dis_url)
    download_file(directory, gen_url)