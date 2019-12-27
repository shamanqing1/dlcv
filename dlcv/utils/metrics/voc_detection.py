
from ignite.metrics import Metric

class VOCMApMetric():
    """
    Calculate mean AP for object detection task

    Parameters:
    ---------
    iou_thresh : float
        IOU overlap threshold for TP
    class_names : list of str
        optional, if provided, will print out AP for each class
    """
    def __init__(self, iou_thresh=0.5, class_names=None):
        super(VOCMApMetric, self).__init__()
        if class_names is None:
            self.num = None
        else:
            assert isinstance(class_names, (list, tuple))
            for name in class_names:
                assert isinstance(name, str), "must provide names as str"
            num = len(class_names)
            self.name = list(class_names) + ['mAP']
            self.num = num + 1
        # self.reset()
        self.iou_thresh = iou_thresh
        self.class_names = class_names


class VOC07MApMetric(VOCMApMetric):
    """ Mean average precision metric for PASCAL V0C 07 dataset

    Parameters:
    ---------
    iou_thresh : float
        IOU overlap threshold for TP
    class_names : list of str
        optional, if provided, will print out AP for each class

    """
    def __init__(self, *args, **kwargs):
        super(VOC07MApMetric, self).__init__(*args, **kwargs)

    def _average_precision(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """

        pass