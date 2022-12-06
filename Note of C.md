# NOTE of C

## 前言

​	此文档将记录一些写下来的较为经典的代码，以及新的语法或是编程的心得。

## 目录


* #### <a href="排序算法" target=''>排序算法</a>

  * ##### <a href="快速排序" target=''>快速排序</a>

  * ##### <a href="归并排序" target=''>归并排序</a>

  * ##### <a href="插入排序" target=''>插入排序</a>

  * ##### <a href="希尔排序" target=''>希尔排序</a>

* #### <a href="数据结构" target=''>数据结构</a>

  * ##### <a href="链表" target=''>链表</a>

  * ##### <a href="栈" target=''>栈</a>

  * ##### <a href="队列" target=''>队列</a>

  * ##### <a href="二叉树" target=''>二叉树</a>

  * ##### <a href="红黑树" target=''>红黑树</a>

## 正文

### 排序算法

#### 快速排序 Quick Sort

##### 介绍

​	通过分界值将数组分为左右两部分，小于分界值的数划分到左半部分，大于等于分界值的数划分到右半部分，依次递归。适用于数据量较低的排序。

##### 代码

```c++
int partition(int *a, int p, int r){
  int tmp, i = p-1;
  for (int j = p; j < r; j++){
    if (a[j]<a[r]){
      i++;
      tmp = a[j];
      a[j] = a[i];
      a[i] = tmp;
    }
  }
  i++;
  tmp = a[i];
  a[i] = a[r];
  a[r] = tmp;
  return i;
}

void quick_sort(int *a, int p, int r){
  if (p >= r) return;
  int q = partition(a, p, r);
  quick_sort(a, p, q-1);
  quick_sort(a, q+1, r);
}
```

#### 归并排序 Merge Sort

##### 介绍

​	运用分治(Divide and Conquer)的思想，将序列一分为二，对有序的子序列进行合并，依次递归。因为需要复制一遍序列，所以空间复杂度较高。

##### 代码

```c++
void merge(int *a, int p, int q, int r){
    int nL = q-p+1;
    int nR = r-q;
    // copy the array
    int *L; L = (int*)malloc(sizeof(int)*nL);
    int *R; R = (int*)malloc(sizeof(int)*nR);
    for (int t = 0; t < nL; t++) L[t] = a[t+p];
    for (int t = 0; t < nR; t++) R[t] = a[t+q+1];
    //merge the array
    int i = 0, j = 0, k = p;
    while (i<nL && j<nR) {
        if (L[i]<R[j]) a[k++] = L[i++];
        else a[k++] = R[j++];
    }
    while (i<nL) a[k++] = L[i++];
    while (j<nR) a[k++] = R[j++];
}

void merge_sort(int *a, int p, int r){
    if (p>=r) return;
    int q = (p+r) / 2;
    merge_sort(a, p, q);
    merge_sort(a, q+1, r);
    merge(a, p, q, r);
}
```



#### 插入排序 Insertion Sort

##### 介绍

​	想像成整理手牌，先抽取一张，然后插入到已经整理好的牌堆里。

##### 代码

```c++
void insert_sort(int *a, int len){
    for(int i = 1; i < len; i++){
        int now = a[i];
        int j = i-1;
        while (j>=0 && a[j]>now){
            a[j+1] = a[j];
            j--;
        }
        a[j+1] = now;
    }
}
```



#### 希尔排序 Shell's Sort

##### 介绍

​	希尔排序类似插入排序，将数组按一定间隔分组后，对每组进行插入排序，直至间隔减至1。默认初始间隔为数组长度的1/2。

##### 代码

```c++
void shell_sort(int *a, int len, int width){
    if (width == 0) return;
    for (int m = 0; m < width; m++){
        for (int i = width+m; i < len; i = i+width){
            int now = a[i];
            int j = i-width;
            while(j>=0 && a[j]>now){
                a[j+width] = a[j];
                j = j - width;
            }
            a[j+width] = now;
        }
    }
    shell_sort(a, len, width/2);
}
```



### 数据结构

#### 链表 Linked List

​	链表中的数据靠指针链接，是一种非连续、非顺序的存储结构。

#### 栈 Stack

​	栈是一种受限制的线性表，遵循**后进先出(LIFO)**原则，只在栈顶进行插入和删除操作。

#### 队列 Queue

​	队列是一种受限制的线性表，遵循**先进先出(FIFO)**原则，只允许在队头进行删除操作，在队尾进行插入操作。

#### 二叉树 Binary Tree

​	字面意思，二叉树每个节点只能有两个子节点，且有左右之分。

#### 红黑树 Red Black Tree

​	红黑树是一种自平衡二叉查找树，可以通过其性质防止二叉树产生倾斜。




