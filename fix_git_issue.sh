cd .git && \
  sudo chgrp -R $(id -g -n ${whoamai}) . &&\
  sudo chmod -R g+rwX . &&\
  sudo find . -type d -exec chmod g+s '{}' + &&\
  cd ..