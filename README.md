# Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration


![visitors](https://visitor-badge.laobi.icu/badge?page_id=joshyZhou/FPro)
[![GitHub Stars](https://img.shields.io/github/stars/joshyZhou/FPro?style=social)](https://github.com/Zheng-MJ/SMFANet) <br>

[Shihao Zhou](https://joshyzhou.github.io/), [Dayu Li](https://github.com/nkldy22), [Jinshan Pan](https://jspan.github.io/), [Juncheng Zhou](https://github.com/ZhouJunCheng99), [Jinglei Shi](https://jingleishi.github.io/) and [Jufeng Yang](https://cv.nankai.edu.cn/)

#### News
- **Jun 26, 2025:** HINT has been accepted to ICCV 2025 :tada: 
<hr />

## Training
### Derain
To train HINT on rain100L, you can run:
```sh
./train.sh Deraining/Options/Deraining_HINT_syn_rain100L.yml
```
### Dehaze
To train HINT on SOTS, you can run:
```sh
./train.sh Dehaze/Options/RealDehazing_HINT.yml
```
### Denoising
To train HINT on WB, you can run:
```sh
./train.sh Denoising/Options/GaussianColorDenoising_HINT.yml
```
### Desnowing
To train HINT on snow100k, you can run:
```sh
./train.sh Desnowing/Options/Desnow_snow100k_HINT.yml
```
### Enhancement 
To train HINT on LOL_v2_real, you can run:
```sh
./train.sh Demoiring/Options/HINT_LOL_v2_real.yml
```

To train HINT on LOL_v2_synthetic, you can run:
```sh
./train.sh Demoiring/Options/HINT_LOL_v2_synthetic.yml
```

## Evaluation
To evaluate HINT, you can refer commands in 'test.sh'

For evaluate on each dataset, you should uncomment corresponding line.


## Results
Experiments are performed for different image processing tasks. 
Here is a summary table containing hyperlinks for easy navigation:
<table>
  <tr>
    <th align="left">Benchmark</th>
    <th align="center">Pretrained model</th>
    <th align="center">Visual Results</th>
  </tr>
  <tr>
    <td align="left">Rain100L</td>
    <td align="center"><a href="https://pan.baidu.com/s/1k93yGwD3m9MF5XwnKXQrOQ?pwd=ngn8">(code:ngn8)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1sgoh9wB78-IL2pH9cUheBw?pwd=bdpg">(code:bdpg)</a></td>
  </tr>
  <tr>
    <td align="left">SOTS</td>
    <td align="center"><a href="https://pan.baidu.com/s/1krrsVUc5rGvQnw5mnsFZXw?pwd=64j8">(code:64j8)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1pQsutMyHQG2rvNIEh1wMnA?pwd=dypf">(code:dypf)</a></td>
  </tr>
  <tr>
    <td align="left">Snow100K</td>
    <td align="center"><a href="https://pan.baidu.com/s/1CnGdJMOKX8Y9VOs7AEApyw?pwd=q2cm">(code:q2cm)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1AyQDD-ST76RyXCqGB2PmJw?pwd=s7xx">(code:s7xx)</a></td>
  </tr>
    <tr>
    <td align="left">LOL-v2-Real</td>
    <td align="center"><a href="https://pan.baidu.com/s/1_f5J7__OW-irltYgRMukxg?pwd=6cux">(code:6cux)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/16QtYQu76YMCInEigKq5V8g?pwd=5bxm">(code:5bxm)</a></td>
  </tr>
  <tr>
    <td align="left">LOL-v2-Syn</td>
    <td align="center"><a href="https://pan.baidu.com/s/1iYkWYpb_ZNoWK6zKh43nwg?pwd=7fi5">(code:7fi5)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1DgB91a-zDccB_7Myr5IH_g?pwd=y9uq">(code:y9uq)</a></td>
  </tr>
  <tr>
    <td align="left">WB</td>
    <td align="center"><a href="https://pan.baidu.com/s/1SHR6ybZ_uZ4YX9XTTjvdZQ?pwd=ah36">(code:7fi5)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1Z04BRs66tNxX9oTExg5yAg?pwd=ss8c">(code:ss8c)</a></td>
  </tr>

</table>


## Citation
If you find this project useful, please consider citing:

    @inproceedings{zhou_ICCV25_HINT,
      title={Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration},
      author={Zhou, Shihao and Li, Dayu and Pan, Jinshan and Zhou, Juncheng and Shi, Jinglei and Yang, Jufeng},
      booktitle={ICCV},
      year={2025}
    }

## Acknowledgement

This code borrows heavily from [Restormer](https://github.com/swz30/Restormer). 