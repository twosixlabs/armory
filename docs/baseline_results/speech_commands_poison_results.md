# Speech Commands Dirty-label Backdoor Baseline Evaluation

**All tables are the mean of 3 runs.**  Results obtained with Armory 0.16.1 December 2022.

Source class: 11

Target class: 2

Note: Because the source class has about 54K examples compared to the 1-3K in the other classes,
we evaluate with lower poison percentages compared to other poison evaluations on more balanced datasets.  After 0 poison, the lowest fraction of poison we test is 0.1%.

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
| 00 |  0.939 | 0.985 | - | - |
| 0.1 |  0.923 | 0.989 | 0.887 | 0.440 |
| 0.5 |  0.930 | 0.983 | 0.856 | 0.905 |
| 01 |  0.941 | 0.985 | 0.862 | 0.955 |
| 05 |  0.940 | 0.989 | 0.859 | 0.983 |
| 10 |  0.950 | 0.981 | 0.869 | 0.995 |


### Random Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.932 | 0.973 | - | - |
| 0.1 |  0.931 | 0.980 | 0.900 | 0.382 |
| 0.5 |  0.931 | 0.980 | 0.858 | 0.885 |
| 01 |  0.909 | 0.972 | 0.833 | 0.933 |
| 05 |  0.930 | 0.984 | 0.850 | 0.979 |
| 10 |  0.941 | 0.974 | 0.876 | 0.812 |


### Perfect Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.918 | 0.993 | - | - |
| 0.1 |  0.931 | 0.979 | 0.931 | 0.001 |
| 0.5 |  0.946 | 0.976 | 0.946 | 0.006 |
| 01 |  0.934 | 0.983 | 0.934 | 0.001 |
| 05 |  0.944 | 0.979 | 0.944 | 0.002 |
| 10 |  0.936 | 0.980 | 0.936 | 0.005 |


### Activation Clustering

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.942 | 0.975 | - | - |
| 0.1 |  0.893 | 0.989 | 0.858 | 0.415 |
| 0.5 |  0.929 | 0.985 | 0.857 | 0.884 |
| 01 |  0.939 | 0.982 | 0.863 | 0.928 |
| 05 |  0.931 | 0.979 | 0.850 | 0.984 |
| 10 |  0.936 | 0.960 | 0.857 | 0.994 |


### Spectral Signatures

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.923 | 0.980 | - | - |
| 0.1 |  0.926 | 0.983 | 0.921 | 0.059 |
| 0.5 |  0.906 | 0.984 | 0.870 | 0.433 |
| 01 |  0.934 | 0.976 | 0.884 | 0.623 |
| 05 |  0.901 | 0.980 | 0.901 | 0.004 |
| 10 |  0.927 | 0.976 | 0.897 | 0.372 |



## Car Horn Trigger

### Undefended

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.910 | 0.916 | - | - |
| 001 |  0.932 | 0.988 | 0.870 | 0.749 |
| 005 |  0.937 | 0.987 | 0.859 | 0.947 |
| 01 |  0.935 | 0.984 | 0.857 | 0.953 |
| 05 |  0.940 | 0.985 | 0.860 | 0.975 |
| 10 |  0.933 | 0.989 | 0.851 | 0.985 |


### Random Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.922 | 0.984 | - | - |
| 001 |  0.928 | 0.981 | 0.878 | 0.618 |
| 005 |  0.922 | 0.960 | 0.847 | 0.932 |
| 01 |  0.945 | 0.973 | 0.867 | 0.958 |
| 05 |  0.929 | 0.987 | 0.849 | 0.967 |
| 10 |  0.933 | 0.984 | 0.854 | 0.973 |


### Perfect Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.948 | 0.987 | - | - |
| 001 |  0.940 | 0.985 | 0.939 | 0.008 |
| 005 |  0.940 | 0.971 | 0.940 | 0.001 |
| 01 |  0.936 | 0.981 | 0.936 | 0.005 |
| 05 |  0.913 | 0.986 | 0.913 | 0.001 |
| 10 |  0.940 | 0.970 | 0.940 | 0.003 |


### Activation Clustering

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.929 | 0.984 | - | - |
| 001 |  0.928 | 0.986 | 0.865 | 0.765 |
| 005 |  0.921 | 0.984 | 0.846 | 0.914 |
| 01 |  0.942 | 0.984 | 0.865 | 0.948 |
| 05 |  0.925 | 0.970 | 0.846 | 0.988 |
| 10 |  0.932 | 0.982 | 0.851 | 0.985 |


### Spectral Signatures

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.935 | 0.988 | - | - |
| 001 |  0.930 | 0.971 | 0.903 | 0.322 |
| 005 |  0.928 | 0.981 | 0.913 | 0.181 |
| 01 |  0.928 | 0.980 | 0.917 | 0.141 |
| 05 |  0.933 | 0.983 | 0.893 | 0.484 |
| 10 |  0.935 | 0.976 | 0.933 | 0.026 |
