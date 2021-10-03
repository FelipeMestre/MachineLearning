import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TitanicDatasetComponent } from './titanic-dataset.component';

describe('TitanicDatasetComponent', () => {
  let component: TitanicDatasetComponent;
  let fixture: ComponentFixture<TitanicDatasetComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ TitanicDatasetComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(TitanicDatasetComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
