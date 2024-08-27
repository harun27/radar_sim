# -*- coding: utf-8 -*-q
"""
Created on Mon Jul  1 19:38:51 2024

@author: harun
"""

from mvp.presenter import Presenter

presenter = Presenter()

while(1):
    if not presenter.run():
        break

