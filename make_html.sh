rm *html
id=1
cat $1 | while read i
do
  cluster=$(echo $i |cut -d ':' -f 2) #grep -o ".$")
  cluster=$(echo $cluster)
	if ! [[ -f "$cluster.html" ]]
		then
            echo "<div><iframe src=\"$cluster.html\" height=200 width=1800></iframe>" >> index.html
			echo "<a href=$cluster.html target='_blank'>Cluster #$cluster</a></div><br>" >> index.html
			echo "<head><title>Cluster $cluster</title></head>" > $cluster.html
			echo "<script type='text/javascript' src='../../../js/src.js'></script>" >> $cluster.html
			echo "<p>      </p>" >> $cluster.html
            #echo "<a href='' id='a'></a><button onclick='download(arr, \"paths.txt\", \"text/plain\")'>Download file</button><br><br>" >> $cluster.html
	fi
	img=$(echo $i | sed "s/.JPG.*/.JPG/" | sed "s/\/home\/mousavi\/da1\/icputrd\/arf\/mean.js\///" )
    #echo "$img"
    src="<img onclick='addToList(this.src, this.id)' src="
    size=" id=\"$id\" height=200 width=200 >"
	echo "$src'../../../../$img'$size" >> $cluster.html
	id=$((id+1))
done 
