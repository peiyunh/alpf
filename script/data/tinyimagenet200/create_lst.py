import os
import random
random.seed(12304)

with open('data/tinyimagenet200/wnids.txt') as f:
    wnid_to_lbl = {l.split()[0]: i for (i, l) in enumerate(f)}

os.chdir('data/tinyimagenet200')

# save png and lst file
train_cnt, val_cnt, test_cnt = 0, 0, 0
imgs = []
with open('train.lst', 'w') as f:
    for wnid, lbl in wnid_to_lbl.iteritems():
        imgdir = os.path.join('./train/%s/images' % wnid)
        imgs.extend([(img, wnid) for img in os.listdir(imgdir)])
    random.shuffle(imgs)
    for img, wnid in imgs:
        imgdir = os.path.join('./train/%s/images' % wnid)
        lbl = wnid_to_lbl[wnid]
        f.write('%d\t%s\t%d\n' % (train_cnt, os.path.join(imgdir, img), lbl))
        train_cnt += 1

with open('./val/val_annotations.txt') as f:
    val_lbls = {l.split()[0]:wnid_to_lbl[l.split()[1]] for l in f}
            
with open('val.lst', 'w') as f:
    imgdir = './val/images'
    imgs = os.listdir(imgdir)
    for img in imgs:
        f.write('%d\t%s\t%d\n' % (val_cnt, os.path.join(imgdir, img), val_lbls[img]))
        val_cnt += 1

with open('test.lst', 'w') as f:
    imgdir = './test/images'
    imgs = os.listdir(imgdir)
    for img in imgs:
        f.write('%d\t%s\t0\n' % (test_cnt, os.path.join(imgdir, img)))
        test_cnt += 1
        
os.chdir('../..')

print('train: %d, val: %d, test: %d' % (train_cnt, val_cnt, test_cnt))
