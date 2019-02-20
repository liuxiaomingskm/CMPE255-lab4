{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "# To support both python 2 and python 3\n",
    "from __future__ import division, print_function, unicode_literals\n",
    "\n",
    "# Common imports\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "# to make this notebook's output stable across runs\n",
    "np.random.seed(42)\n",
    "\n",
    "# To plot pretty figures\n",
    "%matplotlib inline\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt\n",
    "mpl.rc('axes', labelsize=14)\n",
    "mpl.rc('xtick', labelsize=12)\n",
    "mpl.rc('ytick', labelsize=12)\n",
    "\n",
    "# Where to save the figures\n",
    "PROJECT_ROOT_DIR = \"/Users/liuxiaoming/Desktop\"\n",
    "\n",
    "CHAPTER_ID = \"classification\"\n",
    "\n",
    "def save_fig(fig_id, tight_layout=True):\n",
    "    path = os.path.join(PROJECT_ROOT_DIR, \"images\", CHAPTER_ID, fig_id + \".png\")\n",
    "    print(\"Saving figure\", fig_id)\n",
    "    if tight_layout:\n",
    "        plt.tight_layout()\n",
    "    plt.savefig(path, format='png', dpi=300)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sort_by_target(mnist):\n",
    "    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]\n",
    "    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]\n",
    "    mnist.data[:60000] = mnist.data[reorder_train]\n",
    "    mnist.target[:60000] = mnist.target[reorder_train]\n",
    "    mnist.data[60000:] = mnist.data[reorder_test + 60000]\n",
    "    mnist.target[60000:] = mnist.target[reorder_test + 60000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([[0., 0., 0., ..., 0., 0., 0.],\n",
       "        [0., 0., 0., ..., 0., 0., 0.],\n",
       "        [0., 0., 0., ..., 0., 0., 0.],\n",
       "        ...,\n",
       "        [0., 0., 0., ..., 0., 0., 0.],\n",
       "        [0., 0., 0., ..., 0., 0., 0.],\n",
       "        [0., 0., 0., ..., 0., 0., 0.]]),\n",
       " array([0, 0, 0, ..., 9, 9, 9], dtype=int8))"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "try:\n",
    "    from sklearn.datasets import fetch_openml\n",
    "    mnist = fetch_openml('mnist_784', version=1, cache=True)\n",
    "    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings\n",
    "    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset\n",
    "except ImportError:\n",
    "    from sklearn.datasets import fetch_mldata\n",
    "    mnist = fetch_mldata('MNIST original')\n",
    "mnist[\"data\"], mnist[\"target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(70000, 784)"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mnist.data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(70000, 784)"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X, y = mnist[\"data\"], mnist[\"target\"]\n",
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(70000,)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "784"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "28*28"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving figure some_digit_plot\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "some_digit = X[2888]\n",
    "some_digit_image = some_digit.reshape(28, 28)\n",
    "plt.imshow(some_digit_image, cmap = mpl.cm.binary,\n",
    "           interpolation=\"nearest\")\n",
    "plt.axis(\"off\")\n",
    "\n",
    "save_fig(\"some_digit_plot\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_digit(data):\n",
    "    image = data.reshape(28, 28)\n",
    "    plt.imshow(image, cmap = mpl.cm.binary,\n",
    "               interpolation=\"nearest\")\n",
    "    plt.axis(\"off\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "# EXTRA\n",
    "def plot_digits(instances, images_per_row=10, **options):\n",
    "    size = 28\n",
    "    images_per_row = min(len(instances), images_per_row)\n",
    "    images = [instance.reshape(size,size) for instance in instances]\n",
    "    n_rows = (len(instances) - 1) // images_per_row + 1\n",
    "    row_images = []\n",
    "    n_empty = n_rows * images_per_row - len(instances)\n",
    "    images.append(np.zeros((size, size * n_empty)))\n",
    "    for row in range(n_rows):\n",
    "        rimages = images[row * images_per_row : (row + 1) * images_per_row]\n",
    "        row_images.append(np.concatenate(rimages, axis=1))\n",
    "    image = np.concatenate(row_images, axis=0)\n",
    "    plt.imshow(image, cmap = mpl.cm.binary, **options)\n",
    "    plt.axis(\"off\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving figure more_digits_plot\n"
     ]
    },
    {
     "data": {
      "image/png": "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M8A/DM3MzMxsKv8wNDMzMzPAPwzNzMzMbCr/MDQzMzMzwD8MzczMzGwq/zA0MzMzMwCmr/QAzCw/K6ywQnT6mmuuAaB79+6VGo6ZmVUZJ4ZmZmZmBjgxbJW++OILAOabb77ovFNPPRWAQw45BIAOHTrkPzDLzPjx4wF49dVXo/OGDh0K1EZiOGjQIABmnHFGAPbaa69KDseqxPDhwwEYOHAgAE888QQA77//PgCLLrpoRcZlBvDiiy8C0LNnTwDuuOMOADbaaKOKjQmcGJqZmZnZVE4Mm2jixIkA9O3bF4DBgwcDsMcee0TXOf/88wGYc845cx5d/WaaaaY6551yyilAqEHbYost8hxSbu69914AevfuXXD+yJEjgcrvqWXln3/+Keu8lqpPnz4ArL766kDrTAwvuugiAO655x4AHn/88UoOp6KefPJJAHbbbTcAfv75ZwCmmWaaio2psfr16wfAmWeeCYS0E2DttdeuxJAq7ocffgCgffv2AKy88soA3H777dF1FllkkfwH1kQnnHACAN9//32FR1LIiaGZmZmZAU4MG01749qbe+uttwCYddZZAfjmm2+i67Zp0ybn0ZWn2vZO8qTHnkwOVOtRq4nhmDFj6pw3zzzzVGAk2XrjjTcK/gXo0qVLpYaTi3fffReA/v37A4XboNbqk08+AUJS2JLcfffdQEgKta165513ouu0tsRw6623BuDll18uOF/btauuuio676yzzspvYM309ttvV3oIRTkxNDMzMzPAiWHZJk2aBMB9990HhKRQttpqKwBuuummfAfWBI8++milh1AxV199daWHUBGjR4+uc94666xTgZFk68cffwTgp59+qvBI8vP8888DISlcYoklKjmcijn88MOj06W2w3rPzzXXXLmMqSm++uoroG4NcC3VBEOo9548eXKdyzbbbDMgHKG7//77Afjjjz+K3pY+9wDvvfceAIsvvnjB/0XnW2lODM3MzMwM8A9DMzMzM5vKh5Lr8dlnn0WnFW2rQbBavpx33nkA7LjjjjmPrumS0Xpr8N133wGtd+JNrbcuqbXDbI3xwAMPFPxf26rWQq1pLrnkkui85OQyHUJ+7LHH8htYE2nspf6tFZtuumnJy6abbjogtBvSaztu3LiC66233npAaNUU/1up5kPH1brdcmJoZmZmZoATw3rF90DjS4lBWHpr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6Lk58MADKzOwqWrnm8HMzMzMmsU/DM3MzMwMqMFDySeffDIAN954Y53LdN6KK66Y65iqlZafAjj88MOBwhY2Lc2QIUOA+g89vf/++wB88803QMss2K9P8lDj22+/DUCnTp2A0M4GavNwTbEWH+PGjavUcFL30ksvAYWHyseMGVOp4TSZJiCUOkwaP6zYWuk11r9q19NSqdQnOYFKJS3bbLNN7mOqFG2Pd9ttNwAuvfRSIEyc9KFkMzMzM6sKNZMY/vTTTwA888wzBecvuOCC0Wm1qVEhfms3evTo6HQttDMppxnqYostBtReUigLLLAAABMmTABgueWWA0L7Fl1uLZMmjB1zzDHReS1xEpHSob59+xacv/LKKwNhW92aJdPv4cOHR5f169evImNqjk022QSAzz77rOD87bffHqjNIxgN0WS5atPytihmZmZmlomaSQwnTZoEwIsvvlhwfp8+faLTtdrUuCHaE1trrbUAWGKJJQA444wzouvMO++8+Q8sZdtttx0ADz30EFC8SWqttyhKJuatzb777gvAa6+9Fp1XS/Vq+uy2dHPPPTcA66yzDhAasx9wwAFA691WxyVrDOPLXOo9rXr5am7+rdrC5DJwami9++675z6marHBBhsAcN555wGwyCKLVHI4ESeGZmZmZgbANPHZbVWk0YPSLNM111wTgM6dOwOFzY7VBNdq2+233w7ATjvtFJ13xBFHAKHxud8L1pK1a9cuOq36UW37rDY8/PDDAGy66aZAOOIDcMEFFwCw0kor5T+wRtJSrLfeeisQUn3VmRZbttZyU7QtgBNDMzMzMwNqKDE0MzMzs7I5MTQzMzOz0vzD0MzMzMwA/zA0MzMzs6n8w9DMzMzMAP8wNDMzM7Op/MPQzMzMzAD/MDQzMzOzqfzD0MzMzMwA/zA0MzMzs6n8w9DMzMzMAP8wNDMzM7Op/MPQzMzMzACYvtIDqIQddtgBgDvvvBOA9u3bA7DzzjsD0L17dwC23nrr6G9mn332PIeYimuvvRaAm266CQiPu0+fPgAsvvji0XWPO+44APbZZ588h5iJ6667DoBBgwZF52244YYArLPOOgCsvvrqAMw000wAtGnTJs8hWsYeeuih6PQ999wDwJVXXglAv379ADj99NPzH5jl4s8//wRg8uTJQPh8zzbbbBUbk1m5HnnkEQA++ugjANZaa63osummmw4o/P5OmxNDMzMzMwNgmn/++afSYygm00FdeOGFANx6660AvPTSSwBMM800BddbZpllotOHHnooAPvvv3+WQ0vFwIEDgZACau9Z9JrHH+/00/8vPF566aUBuPTSSwFYc801sx1sBoYMGQLAQQcdFJ33ww8/AHUfu5LEE044AQhJIsCMM86Y/WCb4OeffwbgnXfeic7r378/ENIxPU69nmeccQZQmILXkqeeegqAs846C4CHH344ukyvdfK11xGDWn1O5K+//gIKtwPaBv7yyy8ArLbaagD07NkTqLstrAZ630+cOBGAMWPGAPD444/Xue7XX38NwLBhwwBYeOGFgfC+WGKJJTIda0Puv//+6PT8888PwLhx4wCYMmVKwXWLba9l1KhRQPjc77jjjkDYBsoxxxwTnR49ejQATzzxRJPHL08//XTR29LnEODXX38t67YWWWSR6PR//vMfAPbcc8/mDbCFeeGFFwDo1asXAN999x0Q3r8Qvpf0Guu9PPPMMzflLot+0J0YmpmZmRnQShND0Z7MTz/9BMDw4cML/n300Uej6/7+++8AHHDAAQBcfvnleQyxSU488UQAzjnnnKKX17cHKp07dwbCc7HCCiukOcRcvPXWW9Fp7SWfeeaZAHz11VdASBKlR48e0WklrkoRK50gvv322wBss802AIwfPz66LPmaJv+/4IILAiFlAZhrrrkyHnF2lB4pmVBiqse78cYbR9fdaqutAJgwYQIQ3gOqQW0JRwGaQs+RkvObb765wb9RTV7btm2zG1gT7b333gAMHjy4ybeh5OW0004DYLfddmv2uBpD70ttjwBmnXVWIGyTyjnCkwYlyc2hOt2TTz654Pz490X8sxj37rvvAiHVjdNjVc37scceC8Biiy3WzBFXt7322guAG264oey/Of7444GwXWskJ4ZmZmZmVpp/GJqZmZkZ0Erb1YiKNfXvvvvuW/CvCkEBjjzySABuu+02IEwV1/ktUTzijxfrA3z66adAiLRb4qHk+OQhnVYx8zPPPAPAgQceCITDziNHjoz+RqcffPBBoPQhkayNHTsWCAXJX375JVB4aElj02FTXUflBB9//DFQeDjxiCOOyHDU2Sh1OF1tiPT4Dz/88Dp/q0kmet40MadW6H3yySefAOG9/c033wChLRfAUUcdBYQJD5q8oEOsAwYM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dRdQfvF7WmMbMWIEEJZ5LDbZK0ktsrTU3yabbAKEpQ7VED1+hEfJy6mnngoUXxawku6///7otLblejzJREifU6WgAMcddxwA++yzDxCW/IvfLoTvuEp/X2VNz2FWlGxrspYS76b64osvALjvvvsKztf/zznnnLJvS2l4fGJhczkxNDMzMzOghhPDP//8Eyhcwm6nnXYCwq91NXPVv+XUkClp0jJzWtS71ILWaVFN3LXXXguEGiqlnI2hpLBY64YePXoAMNNMMzVpnHlQQqgkQXtXWv6qc+fOQGHNkWjvu5y0U3voWr5KqYPqQ1oSLRWndAXCc5B3fVlzKClUPY0eg9KillgnGada1wkTJhS9XOmA6u0gHAFRDdrjjz9e8Df6POy5557ReSussEI6A67HRx99FJ1W0/Lvvvuu4Doah2p9i/n8888BuP322wG44447gJA+xpuZy6233gqEmtNKU6KlJSnjSenkyZMLrltq2xSvWevTpw8QljYcP358wd9qYYJZZ5212WPPmo7saIlOgOuvv77gOjrKVaolSzwV1+dhtdVWS22MN998M9C0pFDLzL755pupjSfu+++/B8LY9HlvDieGZmZmZgbUYGKoGV0DBw4ECmuOZphhBiAsxK10TInh6aefDoRUIj6T7eyzzwbghRdeAMIezYMPPgiEBDEra621FhBq4ppDydr7779f57JiMxirhZLCY489Fgh7y0naS46nInrelCSXEk8DVaOlfxdeeGEgzHpTTVBLWEZMe+PxdEXpqd5b1UyJ5x577AGEZER78o1phl0ttKcPYXaw9vpL1X3qvR2vmdPnWN0Ukvbaay+g8POgtC1JMxxVo9cUqifcdttto/OUCrVt2xYI9buaOV3fZ0ifWaVGqk9MJoXxOrrmjD9Nl156KQDXXXcd0LQjPMXoOUl+H6jOTLWH1bRtUlqs9PSyyy4DQkr+4osvNngbmvHbrl27gvNnmWWW6HSaSaFo1nyxNFdj0fdSsp5VabmW7SxGvzmGDBnS6LEpSdaRAnVqaA4nhmZmZmYG1FBiqNlXxxxzDBCO58d/vas2RQvRi5bMuvfeewGYeeaZAbjzzjuj62iPXnsjSgzrW/YmDTfddBMQakjqoxmGmqFZSnImVDWbMmVKdFp7VUoKtaem12/77bcHYO211wbCXjqEJdNEM5c1a1Ezj//1r39F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5yFZH6/tihpBx8eWpE4h6higunkdvdD3vjWZE0MzMzMzK63FJ4Zm1jyq5dPyYWeffTZQd/bzX3/9VYHR1Q5ta3WUoxil3S1x9rzVTQz32WcfALp16waEurr40qJpUocN1dw9/PDDQGF3DvV4zJtSPi1H2hjJJW6zPGLRyjgxNDMzM7PSnBiamZmZtT5ODM3MzMysNP8wNDMzMzPAPwzNzMzMbCr/MDQzMzMzwD8MzczMzGwq/zA0MzMzM8A/DM3MzMxsKv8wNDMzMzPAPwzNzMzMbCr/MDQzMzMzwD8MzczMzGwq/zA0MzMzM8A/DM3MzMxsKv8wNDMzMzMApq/0AMzMzFq73377DYBnn30WgGeeeQaAV199FYBhw4ZF11133XUBWGGFFQDo06cPAIsvvnguY7Xa5sTQzMzMzAD/MDQzMzOzqab5559/Kj2GYjIZ1DnnnAPAm2++CcBHH30EQK9evQBYaaWVAOjRo0cWd2+WismTJwNw4403RufdfffdAHz66acAvPfeewCst956QDjktNVWW0V/s9Zaa2U/2JR8/fXXAFx88cUA3HnnnQC88847Ra/ftm3b6LQ+30sssQQAhx12GABzzTVXNoM1a4Rff/0VgL333huAIUOGNPo25p57bgBGjBgBwMorr5zS6Ky5pplmmjrnLbDAAgCsuuqqABx55JEArLbaavkN7H/qDg4nhmZmZmY2VatKDAcMGADAbbfdVnC+0pVffvkFgHnnnReA/fffP7rODjvsAMAyyyyTxdAadM899wDw1VdfAaHY+M8//2zwb5USbbLJJgXnr7jiitHp7t27pzJOy96YMWMA+L//+7/ovHnmmQeAffbZB4AZZpgBgMceewwI7/Evv/wy+hu9p2+66SYApptuuiyH3Wg33HBDdPrggw8G4OeffwagS5cuACy66KJF/1afZYBRo0YVXLbUUksBcNdddwH5f6Z///13AK677rrovN122w0oTDqtdfjwww8B6NatGwDff/99k29r6623BmDo0KFA8bSqUvRbQ9sdpf6dO3cG4IknnoiuW+pz3RJtv/32dc7773//C8ALL7xQcP75558PwFFHHZX9wP7HiaGZmZmZldaqEsNS3n33XQAuvfRSAG6++WagcM+tTZs2QPhFf8ghh+Q5xKgWYfTo0and5vzzz1/ntFLSnj17AtCpU6fU7q8p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IFx5Lr/88jkecf758ssvgeAhi3LfffcB0LNnT6A0FcRff/0VgJ122gkIChHAo48+CsAqq6wS/8CKHO0PUgpFKXgKq4N8O7vuumuBR1K8/PnnnwB89913AKy++urVfg6pV9OmTUss23///XMwuuxp2rQpEI4BxXY8Gz9+fOL2lClTADjyyCMBWHzxxav9fJrB+vHHH4FkhanY0ft+8sknE8vkqUv1/2omcMKECQC0b98eKL7vt6bo91efxXrrrQcEBRyym+GsLVYMjTHGGGMMUIaK4VdffQXA0KFDK9z31FNPASGvbKWVVqr0ufr375+4Ld+a1Ladd9659oONmauuugoIV11RH9aIESMAGD58OFCaV2C6CtcV+NNPP524b/311y/ImIoZbQ8XXHBB0vJtttkGCPtLMSuHqixedtllATj11FMzrquKdBOQv/Tyyy8Hgm/p22+/BWD69OnVfs5bbrkFgNGjRyeW7bHHHgAstdRSNR9sNdD4iw3NaqjaFIK37rDDDgOqpxh++umnADz44INJy3v37l2rccaBjs/67ckGzeLNmDEDgLXXXjv3AysgQ4YMAYLiqzoIeUbjwoqhMcYYY4wBfGJojDHGGGMWUXZTyb/88gsA33//fYX71lprLQBatmxZ6XPosXfeeWeF+0pRupZUf8899wCw+eabA8lRO9WR84uNL774AqhocK9LRQaaElSRjWwCK6ywQtJ6++yzT+J2dKodYIMNNgBg1KhRQHzTftVBU3Ga/n722WeBMN2tMO4oBxxwAACNGzeOY4g54aGHHgLClJIC6HOFPkcV1SkgWBaZqNm9uvz1119ACNKHEC/SqVOnGj9vVSigHUKg9RZbbJG316sJF198MZAczVIbHnnkEQD++OOPpOXFHMc1f/58IMTG6Tc7WlSRGlOjoigVlZYCep8qxvrmm28S98k2IKKFWhDOVVSMEjdWDI0xxhhjDFCGiqFCeWXWVDB1dZD6snDhwtwNrADoSuySSy4BwpX8dtttByQHQOs9q91OKSG18/fffweC0luMile+OP7444Hw3m+++WYATjvtNCCoxVJSIHznujpXZFExf266gk5VtFRYkk4xbNGiBQBLL710nkdXex5//HEAjjvuOAD69OmTs+f+5JNPErdlan/zzTeBcIxQdFdt2gRqzNHvKA7FcNiwYYnbaod51FFH5e31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TjbaaCMAPvvsMwB++eWXCuvuvvvuANx7771A5TEqcSAvy7nnngsEX2EUfU+pKKIl+h3p/SnWpdDIX6OrzKh6Kz/iiSeemPax8rsqAgdg6aWXzss440Dv45RTTgGCQiGlUEHfAE2bNo15dDXnoYceSvr/gAMOAOC+++4rxHDKHimvag8J8NdffwEhwurqq6+Of2AlTtQbm0kplCp42mmnAUFZBFhvvfXyOLr88OWXXwIhaP7dd98FwmyVzh0gzALJJ37RRRcB4TdHvmH9XlXlI841VgyNMcYYYwxgxRAIV4byANxwww1AclPvVOSH2G233TKu065dOyDM+VfWyiyXpFbgDh48GAiqkcauQEwICss666wDhKBNBTXLz6CrOwhKWtxXM6m89957QFAKpXJWpt4ut9xyAOyzzz4A3H777UBxq2gam0Kqo96y6667DoAuXboA0KJFCyBUuR1xxBFAcoW8fHmNGjUCQiWzfGzF/FlcdtllQAjLXWGFFQB46aWXANh4440LM7Ac0bJlS8BqVa6Qr1rbi4536dogSq1KbSlYSmg2SCpW1PP31VdfAcEPr+P3s88+G+MIoWHDhkCY+ZBSCzB//nwAVlpppVjHVBOkiOo3RP7B888/Hwgt7O64447EY5ZffnkgpGPUxg+YD6wYGmOMMcYYAOpl8lwVmLwOSpWpm266KQBff/01kHzFAsl+tP/85z9AaFWl9lrFVOko36PUISmgUgj79u0LwN577w0kNyr/7bffANhkk00AmDNnDhCUGL3fYkRVWfJOVqYUqjXQ0UcfDUDbtm3zPLrcI3/dnnvumVimVnAHH3wwELxp2hakDkcbrEtd0/4g9Jj99tsPgD322CO3b6AWqOWfPLCqxFMOWHVbPxULOvbofUlBiPqS6gI65sqXBWH2Qp+N/laGFLKbbroJCF5wqTdCxzt5viAc66P7SrGhfVbvU+9L7Uk12xXNnRU65mm/VsW72mbmgtdffz1xu2PHjpWu26FDByC0hYMwixH1HaZjm222ASomROQbJV8AnHDCCUCYeZQqHT0+Q/IM5LrrrgsURZV12h9LK4bGGGOMMQaoAx7DaBWmKn90Rq+zfl0ZypelylT5zwC23HJLIChoxYh8GdG8JAjK4GKL/e86QNVgleXuRXPgygH5O6So5StzMA5UgawrfQieqaeeegoICom2efl4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\n",
      "text/plain": [
       "<Figure size 648x648 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(9,9))\n",
    "example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]\n",
    "plot_digits(example_images, images_per_row=10)\n",
    "save_fig(\"more_digits_plot\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y[2888]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "shuffle_index = np.random.permutation(60000)\n",
    "X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_5 = (y_train %2 == 0)\n",
    "y_test_5 = (y_test %2== 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SGDClassifier(alpha=0.0001, average=False, class_weight=None,\n",
       "       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,\n",
       "       l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=5,\n",
       "       n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',\n",
       "       power_t=0.5, random_state=42, shuffle=True, tol=-inf,\n",
       "       validation_fraction=0.1, verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import SGDClassifier\n",
    "\n",
    "sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)\n",
    "sgd_clf.fit(X_train, y_train_5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True])"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sgd_clf.predict([some_digit])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.85785711, 0.7514    , 0.8679934 ])"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring=\"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8578571071446428\n",
      "0.7514\n",
      "0.8679933996699835\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.base import clone\n",
    "\n",
    "skfolds = StratifiedKFold(n_splits=3, random_state=42)\n",
    "\n",
    "for train_index, test_index in skfolds.split(X_train, y_train_5):\n",
    "    clone_clf = clone(sgd_clf)\n",
    "    X_train_folds = X_train[train_index]\n",
    "    y_train_folds = (y_train_5[train_index])\n",
    "    X_test_fold = X_train[test_index]\n",
    "    y_test_fold = (y_train_5[test_index])\n",
    "\n",
    "    clone_clf.fit(X_train_folds, y_train_folds)\n",
    "    y_pred = clone_clf.predict(X_test_fold)\n",
    "    n_correct = sum(y_pred == y_test_fold)\n",
    "    print(n_correct / len(y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.base import BaseEstimator\n",
    "class Never5Classifier(BaseEstimator):\n",
    "    def fit(self, X, y=None):\n",
    "        pass\n",
    "    def predict(self, X):\n",
    "        return np.zeros((len(X), 1), dtype=bool)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.5048, 0.5104, 0.5102])"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "never_5_clf = Never5Classifier()\n",
    "cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring=\"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_predict\n",
    "\n",
    "y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[22296,  8212],\n",
       "       [ 2243, 27249]])"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "confusion_matrix(y_train_5, y_train_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_perfect_predictions = y_train_5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[30508,     0],\n",
       "       [    0, 29492]])"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_train_5, y_train_perfect_predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7684216463156708"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import precision_score, recall_score\n",
    "\n",
    "precision_score(y_train_5, y_train_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7687135020350381"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "4344 / (4344 + 1307)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9239454767394548"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall_score(y_train_5, y_train_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.801328168234643"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "4344 / (4344 + 1077)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8390374578541407"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import f1_score\n",
    "f1_score(y_train_5, y_train_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7846820809248555"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "4344 / (4344 + (1077 + 1307)/2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([220243.92134298])"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_scores = sgd_clf.decision_function([some_digit])\n",
    "y_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True])"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "threshold = 0\n",
    "y_some_digit_pred = (y_scores > threshold)\n",
    "y_some_digit_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True])"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "threshold = 200000\n",
    "y_some_digit_pred = (y_scores > threshold)\n",
    "y_some_digit_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,\n",
    "                             method=\"decision_function\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000,)"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_scores.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "# hack to work around issue #9589 in Scikit-Learn 0.19.0\n",
    "if y_scores.ndim == 2:\n",
    "    y_scores = y_scores[:, 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import precision_recall_curve\n",
    "\n",
    "precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving figure precision_recall_vs_threshold_plot\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):\n",
    "    plt.plot(thresholds, precisions[:-1], \"b--\", label=\"Precision\", linewidth=2)\n",
    "    plt.plot(thresholds, recalls[:-1], \"g-\", label=\"Recall\", linewidth=2)\n",
    "    plt.xlabel(\"Threshold\", fontsize=16)\n",
    "    plt.legend(loc=\"upper left\", fontsize=16)\n",
    "    plt.ylim([0, 1])\n",
    "\n",
    "plt.figure(figsize=(8, 4))\n",
    "plot_precision_recall_vs_threshold(precisions, recalls, thresholds)\n",
    "plt.xlim([-700000, 700000])\n",
    "save_fig(\"precision_recall_vs_threshold_plot\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(y_train_pred == (y_scores > 0)).all()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_pred_90 = (y_scores > 70000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8659205116491548"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision_score(y_train_5, y_train_pred_90)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6993174691016417"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall_score(y_train_5, y_train_pred_90)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving figure precision_vs_recall_plot\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_precision_vs_recall(precisions, recalls):\n",
    "    plt.plot(recalls, precisions, \"b-\", linewidth=2)\n",
    "    plt.xlabel(\"Recall\", fontsize=16)\n",
    "    plt.ylabel(\"Precision\", fontsize=16)\n",
    "    plt.axis([0, 1, 0, 1])\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "plot_precision_vs_recall(precisions, recalls)\n",
    "save_fig(\"precision_vs_recall_plot\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
