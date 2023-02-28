# Speech Commands Dirty-label Backdoor Baseline Evaluation

**All tables are the mean of 3 runs.**  Results obtained with Armory 0.16.6 February 2023.

Source class: 11

Target class: 2

The source class has about 54K examples compared to the 1-3K in the other classes, which may partly explain why the attack success rate is so high even at very low poison percentages.

The following table shows the scale parameters used for each trigger.  These were chosen so that Clapping and Car Horn had equal energy, and Whistle and Dog Clicker had equal energy.

| Trigger | Scale |
|---------|-------|
| Clapping | 0.1 |
| Whistle | 0.1 |
| Car Horn | 0.0903 |
| Dog Clicker | 0.2695 |


## Whistle Trigger

### Undefended

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.928 | 0.989 | - | - |
| 0.1 |  0.942 | 0.989 | 0.873 | 0.839 |
| 0.5 |  0.935 | 0.989 | 0.858 | 0.937 |
| 01 |  0.941 | 0.981 | 0.861 | 0.975 |
| 05 |  0.938 | 0.982 | 0.857 | 0.988 |
| 10 |  0.937 | 0.979 | 0.855 | 0.996 |
| 20 |  0.940 | 0.975 | 0.859 | 0.993 |
| 30 |  0.937 | 0.980 | 0.856 | 0.995 |


### Random Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.925 | 0.984 | - | - |
| 0.1 |  0.928 | 0.983 | 0.867 | 0.755 |
| 0.5 |  0.928 | 0.980 | 0.852 | 0.931 |
| 01 |  0.931 | 0.977 | 0.853 | 0.955 |
| 05 |  0.937 | 0.980 | 0.856 | 0.986 |
| 10 |  0.928 | 0.971 | 0.848 | 0.993 |
| 20 |  0.934 | 0.978 | 0.853 | 0.994 |
| 30 |  0.923 | 0.988 | 0.841 | 0.994 |


### Perfect Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.936 | 0.977 | - | - |
| 0.1 |  0.938 | 0.986 | 0.938 | 0.003 |
| 0.5 |  0.933 | 0.979 | 0.933 | 0.001 |
| 01 |  0.944 | 0.986 | 0.944 | 0.003 |
| 05 |  0.945 | 0.984 | 0.945 | 0.002 |
| 10 |  0.944 | 0.984 | 0.943 | 0.006 |
| 20 |  0.937 | 0.980 | 0.937 | 0.001 |
| 30 |  0.937 | 0.976 | 0.937 | 0.008 |


### Activation Clustering

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.931 | 0.971 | - | - |
| 0.1 |  0.936 | 0.971 | 0.874 | 0.766 |
| 0.5 |  0.936 | 0.986 | 0.857 | 0.960 |
| 01 |  0.934 | 0.972 | 0.854 | 0.982 |
| 05 |  0.939 | 0.986 | 0.858 | 0.987 |
| 10 |  0.942 | 0.986 | 0.860 | 0.992 |
| 20 |  0.937 | 0.988 | 0.855 | 0.995 |
| 30 |  0.949 | 0.980 | 0.868 | 0.994 |


### Spectral Signatures

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.925 | 0.990 | - | - |
| 0.1 |  0.931 | 0.967 | 0.931 | 0.012 |
| 0.5 |  0.938 | 0.980 | 0.913 | 0.310 |
| 01 |  0.844 | 0.982 | 0.815 | 0.328 |
| 05 |  0.933 | 0.979 | 0.898 | 0.417 |
| 10 |  0.913 | 0.977 | 0.906 | 0.087 |
| 20 |  0.934 | 0.982 | 0.865 | 0.850 |
| 30 |  0.925 | 0.982 | 0.871 | 0.662 |



## Clapping Trigger

### Undefended

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.937 | 0.983 | - | - |
| 0.1 |  0.942 | 0.984 | 0.868 | 0.905 |
| 0.5 |  0.937 | 0.989 | 0.857 | 0.973 |
| 01 |  0.922 | 0.982 | 0.842 | 0.983 |
| 05 |  0.943 | 0.989 | 0.860 | 0.995 |
| 10 |  0.932 | 0.977 | 0.851 | 0.997 |
| 20 |  0.944 | 0.983 | 0.862 | 0.999 |
| 30 |  0.942 | 0.983 | 0.860 | 0.999 |


### Random Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.936 | 0.983 | - | - |
| 0.1 |  0.932 | 0.982 | 0.871 | 0.745 |
| 0.5 |  0.929 | 0.989 | 0.849 | 0.970 |
| 01 |  0.919 | 0.980 | 0.840 | 0.963 |
| 05 |  0.921 | 0.981 | 0.839 | 0.995 |
| 10 |  0.948 | 0.985 | 0.866 | 0.992 |
| 20 |  0.940 | 0.974 | 0.859 | 0.996 |
| 30 |  0.934 | 0.985 | 0.852 | 0.998 |


### Perfect Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.932 | 0.985 | - | - |
| 0.1 |  0.936 | 0.961 | 0.936 | 0.011 |
| 0.5 |  0.940 | 0.980 | 0.939 | 0.002 |
| 01 |  0.937 | 0.971 | 0.936 | 0.008 |
| 05 |  0.933 | 0.982 | 0.933 | 0.006 |
| 10 |  0.941 | 0.981 | 0.941 | 0.001 |
| 20 |  0.937 | 0.975 | 0.937 | 0.004 |
| 30 |  0.937 | 0.981 | 0.937 | 0.002 |


### Activation Clustering

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.934 | 0.979 | - | - |
| 0.1 |  0.889 | 0.922 | 0.819 | 0.900 |
| 0.5 |  0.941 | 0.985 | 0.863 | 0.952 |
| 01 |  0.933 | 0.988 | 0.852 | 0.984 |
| 05 |  0.932 | 0.983 | 0.851 | 0.991 |
| 10 |  0.940 | 0.987 | 0.858 | 0.998 |
| 20 |  0.938 | 0.983 | 0.856 | 0.998 |
| 30 |  0.936 | 0.984 | 0.854 | 0.999 |


### Spectral Signatures

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.928 | 0.963 | - | - |
| 0.1 |  0.929 | 0.975 | 0.888 | 0.499 |
| 0.5 |  0.924 | 0.975 | 0.901 | 0.291 |
| 01 |  0.904 | 0.962 | 0.877 | 0.354 |
| 05 |  0.940 | 0.983 | 0.939 | 0.011 |
| 10 |  0.921 | 0.967 | 0.872 | 0.597 |
| 20 |  0.936 | 0.981 | 0.908 | 0.339 |
| 30 |  0.929 | 0.952 | 0.906 | 0.313 |



## Dog Clicker Trigger

### Undefended

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.946 | 0.982 | - | - |
| 0.5 |  0.946 | 0.968 | 0.868 | 0.967 |
| 01 |  0.942 | 0.975 | 0.863 | 0.965 |
| 05 |  0.875 | 0.975 | 0.794 | 0.997 |
| 10 |  0.942 | 0.980 | 0.860 | 1.000 |
| 20 |  0.948 | 0.970 | 0.867 | 1.000 |
| 30 |  0.948 | 0.976 | 0.866 | 1.000 |


### Random Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.933 | 0.983 | - | - |
| 0.5 |  0.927 | 0.981 | 0.848 | 0.962 |
| 01 |  0.927 | 0.983 | 0.847 | 0.971 |
| 05 |  0.904 | 0.984 | 0.822 | 0.996 |
| 10 |  0.923 | 0.988 | 0.841 | 0.999 |
| 20 |  0.934 | 0.983 | 0.852 | 0.998 |
| 30 |  0.926 | 0.969 | 0.845 | 1.000 |


### Perfect Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.934 | 0.989 | - | - |
| 0.5 |  0.936 | 0.978 | 0.935 | 0.011 |
| 01 |  0.921 | 0.989 | 0.921 | 0.005 |
| 05 |  0.941 | 0.984 | 0.940 | 0.005 |
| 10 |  0.949 | 0.985 | 0.948 | 0.004 |
| 20 |  0.943 | 0.987 | 0.942 | 0.003 |
| 30 |  0.943 | 0.986 | 0.943 | 0.002 |


### Activation Clustering

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.924 | 0.990 | - | - |
| 0.5 |  0.943 | 0.982 | 0.864 | 0.967 |
| 01 |  0.948 | 0.979 | 0.867 | 0.993 |
| 05 |  0.937 | 0.981 | 0.855 | 0.997 |
| 10 |  0.933 | 0.979 | 0.851 | 0.999 |
| 20 |  0.942 | 0.978 | 0.860 | 1.000 |
| 30 |  0.935 | 0.987 | 0.853 | 1.000 |


### Spectral Signatures

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.919 | 0.989 | - | - |
| 0.5 |  0.930 | 0.968 | 0.908 | 0.269 |
| 01 |  0.915 | 0.954 | 0.879 | 0.462 |
| 05 |  0.921 | 0.960 | 0.897 | 0.297 |
| 10 |  0.930 | 0.973 | 0.895 | 0.422 |
| 20 |  0.930 | 0.980 | 0.927 | 0.012 |
| 30 |  0.902 | 0.949 | 0.827 | 0.933 |


### DP-Instahide

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.667 | 0.984 | - | - |
| 0.5 |  0.603 | 0.980 | 0.568 | 0.425 |
| 01 |  0.622 | 0.971 | 0.548 | 0.911 |
| 05 |  0.677 | 0.960 | 0.599 | 0.955 |
| 10 |  0.633 | 0.975 | 0.553 | 0.990 |
| 20 |  0.679 | 0.972 | 0.598 | 0.997 |
| 30 |  0.698 | 0.971 | 0.617 | 0.996 |



## Car Horn Trigger

### Undefended

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.925 | 0.989 | - | - |
| 0.5 |  0.920 | 0.975 | 0.843 | 0.941 |
| 01 |  0.941 | 0.983 | 0.863 | 0.949 |
| 05 |  0.938 | 0.984 | 0.858 | 0.971 |
| 10 |  0.933 | 0.980 | 0.853 | 0.984 |
| 20 |  0.942 | 0.982 | 0.861 | 0.990 |
| 30 |  0.941 | 0.969 | 0.860 | 0.998 |


### Random Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.928 | 0.980 | - | - |
| 0.5 |  0.943 | 0.977 | 0.867 | 0.926 |
| 01 |  0.927 | 0.979 | 0.850 | 0.953 |
| 05 |  0.932 | 0.981 | 0.853 | 0.969 |
| 10 |  0.902 | 0.983 | 0.822 | 0.978 |
| 20 |  0.913 | 0.982 | 0.831 | 0.995 |
| 30 |  0.939 | 0.971 | 0.859 | 0.994 |


### Perfect Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.941 | 0.984 | - | - |
| 0.5 |  0.943 | 0.983 | 0.943 | 0.005 |
| 01 |  0.930 | 0.986 | 0.930 | 0.004 |
| 05 |  0.946 | 0.979 | 0.945 | 0.002 |
| 10 |  0.940 | 0.980 | 0.940 | 0.002 |
| 20 |  0.941 | 0.983 | 0.941 | 0.002 |
| 30 |  0.943 | 0.978 | 0.942 | 0.003 |


### Activation Clustering

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.939 | 0.972 | - | - |
| 0.5 |  0.936 | 0.993 | 0.859 | 0.923 |
| 01 |  0.937 | 0.984 | 0.859 | 0.951 |
| 05 |  0.942 | 0.986 | 0.863 | 0.968 |
| 10 |  0.945 | 0.984 | 0.865 | 0.982 |
| 20 |  0.933 | 0.968 | 0.853 | 0.992 |
| 30 |  0.940 | 0.980 | 0.859 | 0.994 |


### Spectral Signatures

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.936 | 0.979 | - | - |
| 0.5 |  0.930 | 0.980 | 0.902 | 0.335 |
| 01 |  0.917 | 0.980 | 0.874 | 0.518 |
| 05 |  0.939 | 0.980 | 0.894 | 0.546 |
| 10 |  0.938 | 0.977 | 0.933 | 0.056 |
| 20 |  0.891 | 0.983 | 0.833 | 0.702 |
| 30 |  0.922 | 0.962 | 0.843 | 0.985 |


### DP-Instahide

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.573 | 0.984 | - | - |
| 0.5 |  0.641 | 0.975 | 0.620 | 0.266 |
| 01 |  0.613 | 0.980 | 0.585 | 0.334 |
| 05 |  0.633 | 0.978 | 0.555 | 0.949 |
| 10 |  0.617 | 0.967 | 0.581 | 0.444 |
| 20 |  0.676 | 0.947 | 0.601 | 0.940 |
| 30 |  0.663 | 0.907 | 0.588 | 0.989 |

