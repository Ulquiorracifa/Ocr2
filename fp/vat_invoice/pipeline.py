import importlib

from . import _special_vat_invoice_pipeline
importlib.reload(_special_vat_invoice_pipeline)
from ._special_vat_invoice_pipeline import SpecialVatInvoicePipeline

from . import _normal_vat_invoice_pipeline
importlib.reload(_normal_vat_invoice_pipeline)
from ._normal_vat_invoice_pipeline import NormalVatInvoicePipeline

from . import _elec_vat_invoice_pipeline
importlib.reload(_elec_vat_invoice_pipeline)
from ._elec_vat_invoice_pipeline import ElecVatInvoicePipeline

class VatInvoicePipeline(object):
    def __init__(self, invoice_type, pars={}, debug=False):
        '''
        Args
            invoice_type [str], can be
                'special'
                'electric'
            pars [dict]: for example, 
                dict(textline_method='simple')
                dict(textline_method='textboxes') (use deeplearning)
        '''
        _pipe = {'special': SpecialVatInvoicePipeline,
                 'normal': NormalVatInvoicePipeline,
                 'elec': ElecVatInvoicePipeline}
        if invoice_type not in _pipe.keys():
            raise NotImplemented
        self.invoice_type = invoice_type
        self.pipe = _pipe[invoice_type](pars=pars, debug=debug)
    
    def __call__(self, image):
        res = self.pipe(image)
        self.__dict__.update(vars(self.pipe))
        return res
        
    def roi_textlines(self, roi_name):
        '''pipe.roi_textlines(roi_name) -> list of rects
        Args
            roi_name [str] support,
                'general', the genereal wireframe,
                'header0', left part of header, containing invoice type,
                'header1', center part of header, containing invoice title,
                'header2', right part of header, containing invoice serial and time,
                'buyer', rect for buyer,
                'tax_free', rect for tax_free price
                'money', rect for money (chinese and digital),
                'saler', rect for saler
        Return
          a list of textline rects, which is inside of specified ROI
        '''
        return self.pipe.roi_textlines(roi_name)

    def predict(self, textline_name):
        '''pipe.predict(textline_name) -> rect
        Args
            textline_name [str] support 
                'type'： invoice type
                'title': invoice title
                'serial': invoice serial number
                'time': time to made invoice
                'tax_free_money': money without tax
        Return
          a textline rect (x, y, w, h)
        '''
        return self.pipe.predict(textline_name)

    def test_port(self, test_info):
        return self.pipe.test_port(test_info)
