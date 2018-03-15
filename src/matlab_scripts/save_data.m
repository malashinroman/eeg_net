function [] = save_data(file, X, Y, y)
varToStr = @(x) inputname(1);
save(file, varToStr(X),varToStr(Y), varToStr(y))
end

