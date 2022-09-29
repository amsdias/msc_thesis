model = GetModel();
type = GetType();
layers = GetLayers();

if(strcmp(model,'ae'))
   eval(['deepauto_' lower(type) '_' num2str(layers)]);
else
   eval(['deepclassify_' lower(type) '_' num2str(layers)]);
end
    



function model = GetModel()
    prompt = 'Which type? Autoencoder (ae)/Classification (cl): ';
    model = input(prompt,'s');
    if(strcmp(model,'ae') == 0 && strcmp(model,'cl') == 0)
        model = GetModel();
    end
end

function type = GetType()
    prompt = 'Which type? Crisp (c)/Fuzzy (f): ';
    type = input(prompt,'s');
    if(strcmp(type,'c') == 0 && strcmp(type,'f') == 0)
        type = GetType();
    end
end

function layers = GetLayers()
    prompt = 'How many layers? 1-5: ';
    [layers,tf] = str2num(input(prompt,'s'));
    %layers = cast(input(prompt,'s'),'uint8');
    if(tf == 0 || (layers < 1 || layers > 5 ))
        layers = GetLayers();
    end
end