while inotifywait -r -e modify,create,delete ./content/; do
    latexmk -pdf -synctex=1 -output-directory=__target main.tex
    # sleep 1
    # cp ./target/main.synctex.gz ./main.synctex.gz
    sleep 3
    cp ./__target/main.pdf ./main.pdf
done
