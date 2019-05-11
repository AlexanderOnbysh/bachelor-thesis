# Installation

## Install PSCyr

1. Download fonts from
[PSCyr.zip](http://blog.harrix.org/wp-content/uploads/2013/02/PSCyr.zip)

2. Edit `install.sh` in fonts' directory

```bash
#!/bin/sh
INSTALLDIR=`kpsewhich -expand-var='$TEXMFLOCAL'`
mkdir -p $INSTALLDIR/{tex/latex,fonts/tfm/public,fonts/vf/public,fonts/type1/public,fonts/map/dvips,fonts/afm/public,doc/fonts}/pscyr
mv dvips/pscyr/* $INSTALLDIR/fonts/map/dvips/pscyr
mv tex/latex/pscyr/* $INSTALLDIR/tex/latex/pscyr
mv fonts/tfm/public/pscyr/* $INSTALLDIR/fonts/tfm/public/pscyr
mv fonts/vf/public/pscyr/* $INSTALLDIR/fonts/vf/public/pscyr
mv fonts/type1/public/pscyr/* $INSTALLDIR/fonts/type1/public/pscyr
mv fonts/afm/public/pscyr/* $INSTALLDIR/fonts/afm/public/pscyr
mv LICENSE doc/README.koi doc/PROBLEMS ChangeLog $INSTALLDIR/doc/fonts/pscyr

mktexlsr

echo "Map pscyr.map\n" >> $INSTALLDIR/web2c/updmap.cfg
updmap-sys
```

4. `chmod +x install.sh && sudo ./install.sh`
